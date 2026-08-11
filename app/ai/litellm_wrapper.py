from typing import Any, AsyncIterator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessageChunk, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
import litellm


def _tools_to_openai_schema(tools: list) -> list[dict]:
    """
    Convert a list of LangChain tools into the OpenAI function-calling schema
    that LiteLLM accepts via the ``tools`` kwarg.
    """
    schemas = []
    for t in tools:
        if hasattr(t, "args_schema") and t.args_schema is not None:
            params = t.args_schema.model_json_schema()
        else:
            params = {"type": "object", "properties": {}, "required": []}
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": params,
            },
        })
    return schemas


class ChatLiteLLM(BaseChatModel):
    """
    Custom LangChain ChatModel implementation wrapper for LiteLLM.
    Ensures seamless compatibility with all LangChain features like .astream() and .invoke(),
    independent of any deprecated or missing langchain_community classes.

    Supports bind_tools() for tool-calling agents (LangChain 1.x compatible).
    """
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    # Stores tool schemas after bind_tools() is called
    _bound_tools: List[dict] = []

    def bind_tools(self, tools: list, **kwargs) -> "ChatLiteLLM":
        """
        Return a copy of this LLM with the given LangChain tools bound.
        The tools are converted to OpenAI-compatible function schemas and
        injected into every subsequent completion call.
        Forces streaming=False for tool-calling rounds to avoid stream aggregation errors.
        """
        bound = ChatLiteLLM(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,  # Tool calls should not be streamed
        )
        bound._bound_tools = _tools_to_openai_schema(tools)
        return bound

    def _messages_to_litellm(self, messages: List[BaseMessage]) -> list[dict]:
        """Convert LangChain BaseMessage list to LiteLLM-compatible dicts."""
        result = []
        for m in messages:
            if m.type in ("system", "system_message"):
                role = "system"
            elif m.type in ("ai", "ai_message", "assistant"):
                role = "assistant"
            elif m.type == "tool":
                role = "tool"
            else:
                role = "user"

            msg: dict = {"role": role, "content": m.content or ""}

            # Preserve tool_call_id for ToolMessages
            if hasattr(m, "tool_call_id") and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id

            # Preserve tool_calls on AIMessages for the history round-trip
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg["tool_calls"] = m.tool_calls

            result.append(msg)
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        litellm_messages = self._messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = dict(
            model=self.model,
            api_key=self.api_key,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        if self._bound_tools:
            call_kwargs["tools"] = self._bound_tools
            call_kwargs["tool_choice"] = "auto"

        response = litellm.completion(**call_kwargs)
        raw = response.choices[0].message

        # Attach tool_calls if the model requested them
        raw_tool_calls = getattr(raw, "tool_calls", None)
        ai_msg = AIMessage(content=raw.content or "")
        if raw_tool_calls:
            import json
            parsed_calls = []
            for tc in raw_tool_calls:
                try:
                    name = tc.function.name
                    args_str = tc.function.arguments
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_calls.append({"name": name, "args": args, "id": tc.id})
                except AttributeError:
                    if isinstance(tc, dict):
                        parsed_calls.append(tc)
            ai_msg.tool_calls = parsed_calls  # type: ignore[attr-defined]

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        litellm_messages = self._messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = dict(
            model=self.model,
            api_key=self.api_key,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        if self._bound_tools:
            call_kwargs["tools"] = self._bound_tools
            call_kwargs["tool_choice"] = "auto"

        response = await litellm.acompletion(**call_kwargs)
        raw = response.choices[0].message

        # Attach tool_calls if the model requested them
        raw_tool_calls = getattr(raw, "tool_calls", None)
        ai_msg = AIMessage(content=raw.content or "")
        if raw_tool_calls:
            import json
            parsed_calls = []
            for tc in raw_tool_calls:
                try:
                    name = tc.function.name
                    args_str = tc.function.arguments
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_calls.append({"name": name, "args": args, "id": tc.id})
                except AttributeError:
                    if isinstance(tc, dict):
                        parsed_calls.append(tc)
            ai_msg.tool_calls = parsed_calls  # type: ignore[attr-defined]

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        litellm_messages = self._messages_to_litellm(messages)
        call_kwargs: dict[str, Any] = dict(
            model=self.model,
            api_key=self.api_key,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs,
        )
        if self._bound_tools:
            call_kwargs["tools"] = self._bound_tools
            call_kwargs["tool_choice"] = "auto"

        response = await litellm.acompletion(**call_kwargs)
        async for chunk in response:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content is not None:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    @property
    def _llm_type(self) -> str:
        return "litellm"
