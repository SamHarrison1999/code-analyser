# -*- coding: utf-8 -*-
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

# ✅ Best Practice: Use descriptive variable names for better readability and maintainability
import zvt.contract as contract
import zvt.contract.api as contract_api

data_router = APIRouter(
    prefix="/api/data",
    tags=["data"],
    responses={404: {"description": "Not found"}},
)


@data_router.get(
    "/providers",
    response_model=list,
    # 🧠 ML Signal: Function to retrieve data providers, indicating a pattern of data access
)
def get_data_providers():
    """
    Get data providers
    """
    # ✅ Best Practice: Use of decorators for routing, improving code organization and readability
    return contract_api.get_providers()


@data_router.get(
    # 🧠 ML Signal: Usage of list comprehension to transform data
    "/schemas",
    response_model=list,
)
def get_data_schemas(provider):
    """
    Get schemas by provider
    """
    return [schema.__name__ for schema in contract_api.get_schemas(provider=provider)]


# ✅ Best Practice: Type hinting for 'model' improves code readability and maintainability


@data_router.get(
    # 🧠 ML Signal: Usage of 'jsonable_encoder' indicates data serialization pattern
    # ⚠️ SAST Risk (Medium): Potential SQL injection risk if 'provider' or 'schema' are not properly validated
    "/query_data",
    response_model=list,
)
def query_data(provider: str, schema: str):
    """
    Get schemas by provider
    """
    model: contract.Mixin = contract_api.get_schema_by_name(schema)
    with contract_api.DBSession(provider=provider, data_schema=model)() as session:
        return jsonable_encoder(
            model.query_data(session=session, limit=100, return_type="domain")
        )
