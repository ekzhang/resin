//! Python protocol adapters; model-format state lives in resin-recipe.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use crate::media_preprocess;
use crate::recipe_events::{self, ParserEvent};
use crate::recipe_media;
use crate::recipe_types::{MediaSources, Message, RenderOptions, Role, Tool};
use numpy::PyArray1;
use pyo3::{
    exceptions::{PyAttributeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict},
};
use pythonize::{depythonize, pythonize};
use resin_recipe::{
    AttributedEvent, AttributedText, Decoder, Encoder, Error, FinishReason, MediaConfig,
    ResponseParser, Tokenizer, TokenizerDecoder, TokenizerEncoder, ToolScope,
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

#[derive(Clone)]
enum TokenizerSource {
    Native(Arc<Tokenizer>, Arc<TokenizerEncoder>),
    Python {
        object: Arc<Py<PyAny>>,
        decode: Arc<Py<PyAny>>,
    },
}

/// A tokenizer shared by rendering, response parsing, and token metadata.
///
/// Fast Hugging Face tokenizers are snapshotted into Rust. Custom Python
/// tokenizers remain supported through their decode method.
#[pyclass(frozen, skip_from_py_object, module = "resin._resin")]
#[derive(Clone)]
pub(crate) struct RecipeTokenizer {
    source: TokenizerSource,
    name_or_path: Option<String>,
}

#[pymethods]
impl RecipeTokenizer {
    #[new]
    pub(crate) fn new(tokenizer: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(prepared) = tokenizer.cast::<Self>() {
            return Ok(prepared.borrow().clone());
        }
        let name_or_path = optional_attr(tokenizer, "name_or_path")?
            .and_then(|value| value.extract::<String>().ok());
        let backend =
            optional_attr(tokenizer, "backend_tokenizer")?.unwrap_or_else(|| tokenizer.clone());
        let serialize = optional_attr(&backend, "to_str")?.filter(|method| method.is_callable());
        let source = if let Some(serialize) = serialize {
            let json: String = serialize.call0()?.extract()?;
            let native = Tokenizer::from_bytes(json.as_bytes())
                .map_err(|error| PyValueError::new_err(format!("invalid tokenizer: {error}")))?;
            {
                let tokenizer = Arc::new(native);
                let encoder =
                    Arc::new(TokenizerEncoder::new(tokenizer.clone()).map_err(recipe_error)?);
                TokenizerSource::Native(tokenizer, encoder)
            }
        } else {
            let decode = tokenizer.getattr("decode")?;
            if !decode.is_callable() {
                return Err(PyValueError::new_err("tokenizer.decode must be callable"));
            }
            TokenizerSource::Python {
                object: Arc::new(tokenizer.clone().unbind()),
                decode: Arc::new(decode.unbind()),
            }
        };
        Ok(Self {
            source,
            name_or_path,
        })
    }

    #[getter]
    fn is_native(&self) -> bool {
        matches!(self.source, TokenizerSource::Native(..))
    }

    #[getter]
    fn name_or_path(&self) -> Option<&str> {
        self.name_or_path.as_deref()
    }

    #[pyo3(signature = (text, *, add_special_tokens = false))]
    fn encode(&self, text: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .encode(text, add_special_tokens)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("add_special_tokens", add_special_tokens)?;
                object
                    .bind(py)
                    .call_method("encode", (text,), Some(&kwargs))?
                    .extract()
            }),
        }
    }

    #[pyo3(signature = (token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, skip_special_tokens = false))]
    fn decode(&self, token_ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .decode(&token_ids, skip_special_tokens)
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { decode, .. } => Python::attach(|py| {
                let kwargs = PyDict::new(py);
                kwargs.set_item("skip_special_tokens", skip_special_tokens)?;
                decode.bind(py).call((token_ids,), Some(&kwargs))?.extract()
            }),
        }
    }

    fn token_to_id(&self, token: &str) -> PyResult<Option<u32>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Ok(tokenizer.token_to_id(token)),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let tokenizer = object.bind(py);
                let method = match optional_attr(tokenizer, "convert_tokens_to_ids")? {
                    Some(method) => method,
                    None => tokenizer.getattr("token_to_id")?,
                };
                method.call1((token,))?.extract()
            }),
        }
    }

    fn id_to_token(&self, token_id: u32) -> PyResult<Option<String>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Ok(tokenizer.id_to_token(token_id)),
            TokenizerSource::Python { object, .. } => Python::attach(|py| {
                let tokenizer = object.bind(py);
                let method = match optional_attr(tokenizer, "convert_ids_to_tokens")? {
                    Some(method) => method,
                    None => tokenizer.getattr("id_to_token")?,
                };
                method.call1((token_id,))?.extract()
            }),
        }
    }

    /// Serialize the prepared native snapshot, for external tokenizer tools.
    fn to_str(&self) -> PyResult<String> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => tokenizer
                .to_string(false)
                .map_err(|error| PyValueError::new_err(error.to_string())),
            TokenizerSource::Python { .. } => Err(PyValueError::new_err(
                "custom Python tokenizer has no serialized backend",
            )),
        }
    }
}

pub(crate) fn optional_attr<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match object.getattr(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.is_instance_of::<PyAttributeError>(object.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

impl RecipeTokenizer {
    pub(crate) fn native(&self) -> Option<Arc<Tokenizer>> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => Some(tokenizer.clone()),
            TokenizerSource::Python { .. } => None,
        }
    }

    fn decoder(&self) -> Result<Box<dyn Decoder>, Error> {
        match &self.source {
            TokenizerSource::Native(tokenizer, _) => {
                Ok(Box::new(TokenizerDecoder::new(tokenizer.clone())?))
            }
            TokenizerSource::Python { decode, .. } => Ok(Box::new(PythonDecoder {
                decode: decode.clone(),
                ids: Vec::new(),
                emitted: String::new(),
            })),
        }
    }
}

/// Compatibility adapter for custom tokenizers without a serialized backend.
/// These retain cumulative decoding; production fast tokenizers use Rust's
/// streaming decoder instead.
struct PythonDecoder {
    decode: Arc<Py<PyAny>>,
    ids: Vec<u32>,
    emitted: String,
}

impl PythonDecoder {
    fn decoded(&self) -> Result<String, Error> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("skip_special_tokens", false)?;
            self.decode
                .bind(py)
                .call((&self.ids,), Some(&kwargs))?
                .extract()
        })
        .map_err(|error: PyErr| Error::new(error.to_string()))
    }

    fn delta(&mut self, text: &str) -> Result<String, Error> {
        let delta = text
            .strip_prefix(&self.emitted)
            .ok_or_else(|| Error::new("decoded output is not prefix-stable"))?
            .to_owned();
        self.emitted = text.to_owned();
        Ok(delta)
    }
}

impl Decoder for PythonDecoder {
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        self.ids.extend_from_slice(ids);
        let text = self.decoded()?;
        self.delta(text.trim_end_matches('\u{fffd}'))
            .map(AttributedText::plain)
    }

    fn finish(&mut self) -> Result<AttributedText, Error> {
        if self.ids.is_empty() {
            return Ok(AttributedText::default());
        }
        let text = self.decoded()?;
        let delta = self.delta(&text)?;
        self.reset();
        Ok(AttributedText::plain(delta))
    }

    fn reset(&mut self) {
        self.ids.clear();
        self.emitted.clear();
    }
}

pub(crate) type SharedParser = Arc<Mutex<Box<dyn ResponseParser>>>;

pub(crate) fn feed(parser: &SharedParser, ids: &[u32]) -> PyResult<Vec<AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .feed(ids)
        .map_err(recipe_error)
}

pub(crate) fn finish(
    parser: &SharedParser,
    reason: FinishReason,
) -> PyResult<Vec<AttributedEvent>> {
    parser
        .lock()
        .map_err(|_| PyValueError::new_err("response parser lock is poisoned"))?
        .finish(reason)
        .map_err(recipe_error)
}

fn parse_finish_reason(reason: &str) -> PyResult<FinishReason> {
    match reason {
        "stop" => Ok(FinishReason::Stop),
        "length" => Ok(FinishReason::Length),
        _ => Err(PyValueError::new_err(
            "finish reason must be 'stop' or 'length'",
        )),
    }
}

fn recipe_error(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

impl RecipeTokenizer {
    fn encoder(&self) -> Arc<dyn Encoder> {
        match &self.source {
            TokenizerSource::Native(_, encoder) => encoder.clone(),
            TokenizerSource::Python { object, .. } => Arc::new(PythonEncoder(object.clone())),
        }
    }
}
struct PythonEncoder(Arc<Py<PyAny>>);
impl PythonEncoder {
    fn raw(tokenizer: &Bound<'_, PyAny>, text: &str) -> PyResult<Vec<u32>> {
        let kwargs = PyDict::new(tokenizer.py());
        kwargs.set_item("add_special_tokens", false)?;
        tokenizer
            .call_method("encode", (text,), Some(&kwargs))?
            .extract()
    }
}
impl Encoder for PythonEncoder {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>, Error> {
        Python::attach(|py| {
            let tokenizer = self.0.bind(py);
            if let Some(ordinary) = optional_attr(tokenizer, "encode_ordinary")? {
                ordinary.call1((text,))?.extract()
            } else {
                Self::raw(tokenizer, text)
            }
        })
        .map_err(|error: PyErr| Error::new(error.to_string()))
    }
    fn special_id(&self, spelling: &str) -> Result<u32, Error> {
        Python::attach(|py| {
            let tokenizer = self.0.bind(py);
            let value = if let Some(method) = optional_attr(tokenizer, "convert_tokens_to_ids")? {
                method.call1((spelling,))?
            } else {
                tokenizer.call_method1("token_to_id", (spelling,))?
            };
            if value.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err("control token IDs must be integers"));
            }
            let id: u32 = value.extract()?;
            let inverse = match optional_attr(tokenizer, "convert_ids_to_tokens")? {
                Some(inverse) => Some(inverse),
                None => optional_attr(tokenizer, "id_to_token")?,
            };
            if let Some(inverse) = inverse {
                let round_trip: String = inverse.call1((id,))?.extract()?;
                if round_trip != spelling {
                    return Err(PyValueError::new_err("control token did not round-trip"));
                }
            }
            Ok(id)
        })
        .map_err(|error: PyErr| {
            Error::new(format!("missing model control token {spelling:?}: {error}"))
        })
    }
}

/// Common native handle recognized directly by the HTTP actor.
#[pyclass(name = "_NativeResponseParser", frozen, module = "resin._resin")]
pub(crate) struct NativeResponseParser {
    pub(crate) inner: SharedParser,
}
#[pymethods]
impl NativeResponseParser {
    #[pyo3(signature = (token_ids: "Sequence[int] | npt.NDArray[np.uint32]"))]
    fn feed(&self, py: Python<'_>, token_ids: Vec<u32>) -> PyResult<Vec<Py<ParserEvent>>> {
        let events = py.detach(|| feed(&self.inner, &token_ids))?;
        recipe_events::to_python(py, events)
    }

    #[pyo3(signature = (reason: r#"Literal["stop", "length"]"#))]
    fn finish(&self, py: Python<'_>, reason: &str) -> PyResult<Vec<Py<ParserEvent>>> {
        let reason = parse_finish_reason(reason)?;
        let events = py.detach(|| finish(&self.inner, reason))?;
        recipe_events::to_python(py, events)
    }
}

/// Rendered IDs and optional prepared media. Native parser state contains no pixels.
#[pyclass(frozen, module = "resin._resin")]
pub(crate) struct RenderedPrompt {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) media: Option<Py<PyAny>>,
    seed: Option<ParserSeed>,
}
struct ParserSeed {
    recipe_id: u64,
    request: Value,
}
#[pymethods]
impl RenderedPrompt {
    #[new]
    #[pyo3(signature=(token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, media: "MediaInput | None" = None))]
    fn new(token_ids: Vec<u32>, media: Option<Py<PyAny>>) -> Self {
        Self {
            token_ids,
            media,
            seed: None,
        }
    }
    #[getter]
    fn token_ids<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        crate::arrays::readonly_view(this.clone().into_any(), &this.get().token_ids)
    }
    #[getter]
    fn media(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.media.as_ref().map(|m| m.clone_ref(py))
    }
}

/// A native model recipe constructed by get_native_recipe.
#[pyclass(frozen, module = "resin._resin", skip_from_py_object)]
pub(crate) struct NativeRecipe {
    tokenizer: RecipeTokenizer,
    encoder: Arc<dyn Encoder>,
    recipe: Box<dyn resin_recipe::Recipe>,
    id: u64,
    stop_token_ids: OnceLock<Vec<u32>>,
    media_config: Option<MediaConfig>,
    media_preparation_cache: Mutex<media_preprocess::PreparationCache>,
}
impl NativeRecipe {
    pub(crate) fn resolved_stop_token_ids(&self) -> PyResult<&[u32]> {
        if let Some(ids) = self.stop_token_ids.get() {
            return Ok(ids);
        }
        let ids = self
            .recipe
            .stop_token_ids(self.encoder.as_ref())
            .map_err(recipe_error)?;
        // Resolve lazily so template inspection does not require a complete
        // tokenizer. Once published, the backing vector never changes.
        let _ = self.stop_token_ids.set(ids);
        Ok(self.stop_token_ids.get().unwrap())
    }
    pub(crate) fn message_from_wire(&self, py: Python<'_>, value: &Value) -> PyResult<Message> {
        let mut message = Message::from_wire(py, value)?;
        message.lower_wire_fields(&self.recipe.message_policy());
        Ok(message)
    }
    fn request(
        &self,
        py: Python<'_>,
        messages: &[Py<Message>],
        tools: &[Py<Tool>],
        options: Option<&RenderOptions>,
    ) -> PyResult<(Value, MediaSources)> {
        let mut values = Vec::new();
        let mut sources = Vec::new();
        let mut definitions = tools
            .iter()
            .map(|t| t.borrow(py).declaration())
            .collect::<Vec<_>>();
        for (index, message) in messages.iter().enumerate() {
            let message = message.borrow(py);
            message.validate_for(&self.recipe.message_policy())?;
            let mut rendered = message.render_values(py, &mut sources)?;
            if !message.tools.is_empty() {
                if !matches!(message.role, Role::SYSTEM | Role::DEVELOPER) {
                    return Err(PyValueError::new_err(
                        "tool declarations belong to instruction messages",
                    ));
                }
                if matches!(
                    self.recipe.message_policy().tools,
                    ToolScope::InitialInstruction
                ) {
                    if index != 0 {
                        return Err(PyValueError::new_err(
                            "this recipe supports tool declarations only in the initial instruction message",
                        ));
                    }
                    if !definitions.is_empty() {
                        return Err(PyValueError::new_err(
                            "pass tools either on the initial message or to render, not both",
                        ));
                    }
                    definitions = message.tools.iter().map(Tool::declaration).collect();
                    for item in &mut rendered {
                        item.as_object_mut().unwrap().remove("tools");
                    }
                }
            }
            values.extend(rendered);
        }
        let extra = options.map(RenderOptions::to_map).unwrap_or_default();
        Ok((
            json!({"messages":values,"tools":definitions,"extra":extra}),
            sources,
        ))
    }
    fn render_request(
        &self,
        py: Python<'_>,
        request: Value,
        sources: MediaSources,
    ) -> PyResult<RenderedPrompt> {
        // Native tokenizer/processor configurations are immutable snapshots.
        // Python fallback tokenizers may change their special IDs between calls.
        let key = if self.tokenizer.is_native() {
            media_preprocess::cache_key(py, &sources)?
        } else {
            None
        };
        let cached = key.and_then(|key| {
            self.media_preparation_cache
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .get(key)
        });
        // This lookup precedes image decoding as well as resizing/patchifying.
        let sources = if cached.is_some() {
            Vec::new()
        } else {
            media_preprocess::sources(py, sources)?
        };
        let (token_ids, preparation, placements) = py.detach(|| -> PyResult<_> {
            let preparation = if let Some(prepared) = cached {
                Some(prepared)
            } else if sources.is_empty() {
                None
            } else {
                let config = self
                    .media_config
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("this recipe does not support media"))?;
                let prepared = Arc::new(
                    media_preprocess::process(config, sources, self.encoder.as_ref())
                        .map_err(recipe_error)?,
                );
                if let Some(key) = key {
                    self.media_preparation_cache
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner())
                        .insert(key, prepared.clone());
                }
                Some(prepared)
            };
            let prompt = self.recipe.render(&request).map_err(recipe_error)?;
            let expansions = preparation
                .as_ref()
                .map(|p| p.expansions.as_slice())
                .unwrap_or(&[]);
            let (ids, placements) = recipe_media::encode(prompt, expansions, self.encoder.as_ref())
                .map_err(recipe_error)?;
            // Each bound payload owns independent read-only NumPy buffers. Clone
            // cached tensors outside the GIL; oversized/uncached values move.
            Ok((ids, preparation.map(Arc::unwrap_or_clone), placements))
        })?;
        let media = preparation
            .map(|p| p.bind(py, &token_ids, &placements))
            .transpose()?;
        // Parser creation needs controls, schemas and IDs, never images or conversation history.
        let seed = ParserSeed {
            recipe_id: self.id,
            request: json!({"tools":request["tools"],"extra":request["extra"]}),
        };
        Ok(RenderedPrompt {
            token_ids,
            media,
            seed: Some(seed),
        })
    }
}
#[pymethods]
impl NativeRecipe {
    fn _media_preparation_cache_info(&self) -> HashMap<&'static str, usize> {
        self.media_preparation_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .info()
            .into_iter()
            .collect()
    }

    #[getter]
    fn tokenizer(&self) -> RecipeTokenizer {
        self.tokenizer.clone()
    }
    #[pyo3(signature=(messages: "Sequence[Message]", *, tools: "Sequence[Tool]" = Vec::new(), options=None))]
    pub(crate) fn render(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<RenderedPrompt> {
        let (request, sources) = self.request(py, &messages, &tools, options)?;
        self.render_request(py, request, sources)
    }
    #[pyo3(signature=(messages: "Sequence[Message]", *, tools: "Sequence[Tool]" = Vec::new(), options=None))]
    /// Inspect the template before tokenization and media expansion.
    fn debug_render_text(
        &self,
        py: Python<'_>,
        messages: Vec<Py<Message>>,
        tools: Vec<Py<Tool>>,
        options: Option<&RenderOptions>,
    ) -> PyResult<String> {
        let (request, _) = self.request(py, &messages, &tools, options)?;
        py.detach(|| {
            self.recipe
                .render(&request)
                .map(|prompt| prompt.rendered_text())
                .map_err(recipe_error)
        })
    }
    /// Static model stop tokens, returned as an owned list.
    #[getter]
    fn stop_token_ids(&self) -> PyResult<Vec<u32>> {
        Ok(self.resolved_stop_token_ids()?.to_vec())
    }
    pub(crate) fn create_parser(
        &self,
        py: Python<'_>,
        prompt: &RenderedPrompt,
    ) -> PyResult<NativeResponseParser> {
        let seed = prompt
            .seed
            .as_ref()
            .filter(|seed| seed.recipe_id == self.id)
            .ok_or_else(|| {
                PyValueError::new_err("prompt must have been rendered by this recipe instance")
            })?;
        let parser = py.detach(|| {
            self.recipe
                .create_parser(
                    self.encoder.as_ref(),
                    self.tokenizer.decoder().map_err(recipe_error)?,
                    &seed.request,
                    &prompt.token_ids,
                )
                .map_err(recipe_error)
        })?;
        Ok(NativeResponseParser {
            inner: Arc::new(Mutex::new(parser)),
        })
    }
    #[pyo3(signature=(token_ids: "Sequence[int] | npt.NDArray[np.uint32]", *, prompt))]
    fn parse_response(
        &self,
        py: Python<'_>,
        token_ids: Vec<u32>,
        prompt: &RenderedPrompt,
    ) -> PyResult<Py<PyAny>> {
        let parser = self.create_parser(py, prompt)?;
        let events = py.detach(|| {
            let mut events = feed(&parser.inner, &token_ids)?;
            events.extend(finish(&parser.inner, FinishReason::Stop)?);
            Ok::<_, PyErr>(events)
        })?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::<Value>::new();
        let mut indices = HashMap::new();
        for event in events {
            match event.event {
                resin_recipe::Event::TextDelta { delta } => content.push_str(&delta),
                resin_recipe::Event::ReasoningDelta { delta } => reasoning.push_str(&delta),
                resin_recipe::Event::ToolCallStart { index, id, name } => {
                    indices.insert(index, calls.len());
                    calls.push(json!({"id":id,"name":name,"arguments":""}));
                }
                resin_recipe::Event::ToolCallArgumentsDelta { index, delta } => {
                    if let Some(index) = indices.get(&index) {
                        let previous = calls[*index]["arguments"].as_str().unwrap_or("");
                        calls[*index]["arguments"] = json!(format!("{previous}{delta}"));
                    }
                }
            }
        }
        Ok(pythonize(
            py,
            &json!({"content":content,"reasoning_content":reasoning,"tool_calls":calls}),
        )?
        .unbind())
    }
}

fn recipe_params<T: DeserializeOwned>(params: Map<String, Value>) -> PyResult<T> {
    serde_json::from_value(Value::Object(params))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn image_processor_params(
    owner: &Bound<'_, PyAny>,
    mut params: Map<String, Value>,
) -> PyResult<Map<String, Value>> {
    if params.get("media_config").is_none_or(Value::is_null) {
        params.insert(
            "media_config".into(),
            media_preprocess::config_of(owner, "image_processor")?,
        );
    }
    Ok(params)
}

/// Construct a native recipe from a canonical name and typed family parameters.
#[pyfunction]
#[pyo3(signature = (name, tokenizer, *, stop_token_ids: "Sequence[int] | None" = None, **kwargs))]
pub(crate) fn get_native_recipe(
    py: Python<'_>,
    name: &str,
    tokenizer: &Bound<'_, PyAny>,
    stop_token_ids: Option<Vec<u32>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<NativeRecipe> {
    let mut params = kwargs
        .map(|kwargs| depythonize::<Map<String, Value>>(kwargs))
        .transpose()?
        .unwrap_or_default();
    let recipe: Box<dyn resin_recipe::Recipe> = match name {
        "gpt_oss" => {
            if params
                .get("conversation_start_date")
                .is_none_or(Value::is_null)
            {
                let date: String = py
                    .import("datetime")?
                    .getattr("date")?
                    .call_method0("today")?
                    .call_method0("isoformat")?
                    .extract()?;
                params.insert("conversation_start_date".into(), Value::String(date));
            }
            Box::new(resin_recipe::gpt_oss::GptOssRecipe::new(recipe_params(
                params,
            )?))
        }
        "deepseek_v41" => Box::new(
            resin_recipe::deepseek::DeepSeekV41Recipe::new(recipe_params(params)?)
                .map_err(recipe_error)?,
        ),
        "glm5_3_flash" => Box::new(resin_recipe::glm::Glm53FlashRecipe::new(recipe_params(
            image_processor_params(tokenizer, params)?,
        )?)),
        "gemma4" => {
            if params.get("media_config").is_none_or(Value::is_null) {
                params.insert(
                    "media_config".into(),
                    json!({
                        "image_processor": media_preprocess::config_of(tokenizer, "image_processor")?,
                        "video_processor": media_preprocess::config_of(tokenizer, "video_processor")?,
                    }),
                );
            }
            Box::new(resin_recipe::gemma::Gemma4Recipe::new(recipe_params(
                params,
            )?))
        }
        "inkling" => Box::new(resin_recipe::inkling::InklingRecipe::new(recipe_params(
            image_processor_params(tokenizer, params)?,
        )?)),
        "lfm2" => Box::new(resin_recipe::lfm2::Lfm2Recipe::new(recipe_params(params)?)),
        "qwen3_5" => Box::new(resin_recipe::qwen::Qwen35Recipe::new(recipe_params(
            image_processor_params(tokenizer, params)?,
        )?)),
        "qwen3_8" => Box::new(resin_recipe::qwen::Qwen38Recipe::new(recipe_params(
            image_processor_params(tokenizer, params)?,
        )?)),
        "ornith" => Box::new(resin_recipe::qwen::OrnithRecipe::new(recipe_params(
            image_processor_params(tokenizer, params)?,
        )?)),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown native recipe: {name}"
            )));
        }
    };
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let original_tokenizer =
        optional_attr(tokenizer, "tokenizer")?.unwrap_or_else(|| tokenizer.clone());
    let tokenizer = RecipeTokenizer::new(&original_tokenizer)?;
    let media_config = recipe.media_config();
    Ok(NativeRecipe {
        encoder: tokenizer.encoder(),
        tokenizer,
        recipe,
        id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        stop_token_ids: stop_token_ids.map(OnceLock::from).unwrap_or_default(),
        media_config,
        media_preparation_cache: Mutex::default(),
    })
}
