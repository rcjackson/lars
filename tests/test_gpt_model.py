import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_openai():
    with patch("lars.nepho.models.gpt_model.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_chat_returns_content(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test response"
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    model = GPTModel(model_name="gpt-4", api_key="test-key")
    result = await model.chat("Hello")

    assert result == "Test response"


@pytest.mark.asyncio
async def test_chat_passes_temperature(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Response"
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    model = GPTModel(model_name="gpt-4", api_key="test-key")
    await model.chat("Hello")

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_chat_raises_on_api_error(mock_openai):
    from lars.nepho.models.gpt_model import GPTModel

    mock_openai.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

    model = GPTModel(model_name="gpt-4", api_key="test-key")

    with pytest.raises(RuntimeError, match="Error calling GPT API"):
        await model.chat("Hello")
