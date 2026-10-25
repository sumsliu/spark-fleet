# SparkFleet reference overlay: modified deployment/latency behavior; see overlays/README.md and NOTICE.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus
from vllm.entrypoints.anthropic.compat import error_type, error_message

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def messages(request: Request) -> AnthropicServingMessages:
    return request.app.state.anthropic_serving_messages


def translate_error_response(response: ErrorResponse) -> JSONResponse:
    anthropic_error = AnthropicErrorResponse(
        error=AnthropicError(
            type=error_type(response.error.type, response.error.code),
            message=error_message(response.error.message),
        )
    )
    return JSONResponse(
        status_code=response.error.code, content=anthropic_error.model_dump()
    )


@router.post(
    "/v1/messages",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest, raw_request: Request):
    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.serving_tokenization
        error = base_server.create_error_response(
            NotImplementedError("The model does not support Messages API")
        )
        return translate_error_response(error)

    try:
        generator = await handler.create_messages(request, raw_request)
    except Exception as e:
        logger.exception("Error in create_messages: %s", e)
        return translate_error_response(create_error_response(e))

    if isinstance(generator, ErrorResponse):
        return translate_error_response(generator)

    elif isinstance(generator, AnthropicMessagesResponse):
        resp = generator.model_dump(exclude_none=True)
        resp['stop_sequence'] = generator.stop_sequence
        logger.debug("Anthropic Messages Response: %s", resp)
        return JSONResponse(content=resp)

    from vllm.native_ttl_ext.ttl_api import streaming_response
    return streaming_response(generator)


@router.post(
    "/v1/messages/count_tokens",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": AnthropicCountTokensResponse},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def count_tokens(request: AnthropicCountTokensRequest, raw_request: Request):
    handler = messages(raw_request)
    if handler is None:
        base_server = raw_request.app.state.serving_tokenization
        error = base_server.create_error_response(
            NotImplementedError("The model does not support Messages API")
        )
        return translate_error_response(error)

    try:
        response = await handler.count_tokens(request, raw_request)
    except Exception as e:
        logger.exception("Error in count_tokens: %s", e)
        return translate_error_response(create_error_response(e))

    if isinstance(response, ErrorResponse):
        return translate_error_response(response)

    return JSONResponse(content=response.model_dump(exclude_none=True))


def attach_router(app: FastAPI):
    from fastapi.exceptions import RequestValidationError
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.exception_handlers import http_exception_handler
    from starlette.exceptions import HTTPException
    previous = app.exception_handlers.get(RequestValidationError, request_validation_exception_handler)
    async def anthropic_validation(request, exc):
        if request.url.path not in {'/v1/messages', '/v1/messages/count_tokens'}:
            return await previous(request, exc)
        details = '; '.join(
            '.'.join(str(x) for x in e.get('loc', ()) if x != 'body') + ': ' + e.get('msg', 'Invalid request')
            if isinstance(e, dict) else str(e) for e in exc.errors())
        return JSONResponse(status_code=400, content={'type': 'error',
            'error': {'type': 'invalid_request_error', 'message': details}})
    app.add_exception_handler(RequestValidationError, anthropic_validation)
    previous_http = app.exception_handlers.get(HTTPException, http_exception_handler)
    async def anthropic_http_error(request, exc):
        if request.url.path not in {'/v1/messages', '/v1/messages/count_tokens'}:
            return await previous_http(request, exc)
        detail = exc.detail
        if isinstance(detail, dict):
            inner = detail.get('error', detail)
            message = inner.get('message', str(detail)) if isinstance(inner, dict) else str(inner)
        else:
            message = str(detail)
        return JSONResponse(status_code=exc.status_code, headers=exc.headers,
            content={'type':'error','error':{'type':error_type('', exc.status_code),
                                           'message':error_message(message)}})
    app.add_exception_handler(HTTPException, anthropic_http_error)
    app.include_router(router)


@router.get("/v1/native_ttl/capabilities")
async def native_ttl_capabilities(raw_request: Request):
    from vllm.native_ttl_ext.ttl_runtime import config
    cfg = config()
    handler = messages(raw_request)
    if cfg is None or handler is None:
        return {"schema": 2, "enabled": False}
    result = await handler.engine_client.engine_core.call_utility_async("native_ttl_capabilities")
    if result.get("identity") != cfg["identity"]:
        return {"schema": 2, "enabled": False}
    return result
