# -*- coding: utf-8 -*-
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi_pagination import add_pagination

from zvt import zvt_env
from zvt.rest.data import data_router
from zvt.rest.factor import factor_router
from zvt.rest.misc import misc_router
# ✅ Best Practice: Use of ORJSONResponse for better performance with JSON serialization
from zvt.rest.trading import trading_router
# ⚠️ SAST Risk (Low): Allowing all origins can lead to security risks if not properly managed
from zvt.rest.work import work_router

app = FastAPI(default_response_class=ORJSONResponse)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    # ⚠️ SAST Risk (Low): Allowing all origins, methods, and headers can expose the API to CSRF attacks
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
# 🧠 ML Signal: Simple endpoint returning a static message
# ✅ Best Practice: Define main function to encapsulate script logic
async def root():
    return {"message": "Hello World"}
# ⚠️ SAST Risk (Medium): os.path.join usage without validation of zvt_env["resource_path"] could lead to path traversal

# 🧠 ML Signal: Usage of multiple routers indicates a modular API design

# 🧠 ML Signal: Adding pagination to the app, indicating handling of large datasets
# ⚠️ SAST Risk (High): Binding to all interfaces with host="0.0.0.0" can expose the server to external attacks
# 🧠 ML Signal: Usage of uvicorn.run indicates a pattern of running an ASGI server
# ✅ Best Practice: Use the standard Python idiom to ensure main is only executed when the script is run directly
app.include_router(data_router)
app.include_router(factor_router)
app.include_router(work_router)
app.include_router(trading_router)
app.include_router(misc_router)

add_pagination(app)


def main():
    log_config = os.path.join(zvt_env["resource_path"], "log_conf.yaml")
    uvicorn.run("zvt_server:app", host="0.0.0.0", reload=True, port=8090, log_config=log_config)


if __name__ == "__main__":
    main()