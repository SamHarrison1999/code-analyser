# -*- coding: utf-8 -*-
import json

# ✅ Best Practice: Group imports into standard library, third-party, and local sections for better readability.
from typing import Type, List

from zvt.contract.api import del_data, get_db_session

# ✅ Best Practice: Class docstring provides a description of the class purpose
from zvt.contract.zvt_info import StateMixin
from zvt.utils.str_utils import to_snake_str


class StatefulService(object):
    """
    Base service with state could be stored in state_schema
    # ✅ Best Practice: Initializing class variables to None for clarity
    # ⚠️ SAST Risk (Low): Use of assert statement for runtime checks can be disabled with optimization flags.
    """

    # 🧠 ML Signal: Checking if an attribute is None before assigning a default value.
    #: state schema
    state_schema: Type[StateMixin] = None
    # 🧠 ML Signal: Converting class name to snake case string for naming consistency.

    # 🧠 ML Signal: Initializing a database session with specific schema and provider.
    #: name of the service, default name of class if not set manually
    name = None

    def __init__(self) -> None:
        # ✅ Best Practice: Use a list comprehension or extend method for better readability when adding to lists
        assert self.state_schema is not None
        if self.name is None:
            self.name = to_snake_str(type(self).__name__)
        # ✅ Best Practice: Use list concatenation or extend method for better readability
        self.state_session = get_db_session(
            data_schema=self.state_schema, provider="zvt"
        )

    # ⚠️ SAST Risk (Medium): Ensure that del_data function properly sanitizes inputs to prevent SQL injection
    def clear_state_data(self, entity_id=None):
        """
        clear state of the entity

        :param entity_id: entity id
        # ⚠️ SAST Risk (Medium): Using json.loads with a custom object_hook can lead to code execution if the input is not trusted.
        """
        # ✅ Best Practice: Ensure that the state input is sanitized or comes from a trusted source to prevent security risks.
        filters = [self.state_schema.state_name == self.name]
        if entity_id:
            filters = filters + [self.state_schema.entity_id == entity_id]
        del_data(self.state_schema, filters=filters)

    def decode_state(self, state: str):
        """
        decode state

        :param state:
        :return:
        # ✅ Best Practice: Returning None explicitly is clear, but consider if this is the intended behavior or a placeholder
        """

        return json.loads(state, object_hook=self.state_object_hook())

    # ✅ Best Practice: Consider adding a constructor to initialize the state object
    def encode_state(self, state: object):
        """
        encode state

        :param state:
        :return:
        # ✅ Best Practice: Type hinting for 'self.state' improves code clarity and maintainability.
        """
        # 🧠 ML Signal: Method for persisting state, indicating a pattern of saving or updating data

        return json.dumps(state, cls=self.state_encoder())

    # 🧠 ML Signal: Handling of NoneType for missing or invalid data.
    # 🧠 ML Signal: Encoding state before persistence, common in data processing

    def state_object_hook(self):
        # ⚠️ SAST Risk (Low): Potential issue if self.state_domain is not properly initialized elsewhere
        return None

    # 🧠 ML Signal: Lazy initialization of state_domain, a pattern for resource management
    def state_encoder(self):
        return None


# 🧠 ML Signal: Assigning encoded state to domain object, a pattern in ORM usage


class OneStateService(StatefulService):
    """
    StatefulService which saving all states in one object
    """

    # 🧠 ML Signal: Usage of query_data method with filters and entity_ids

    def __init__(self) -> None:
        super().__init__()
        self.state_domain = self.state_schema.get_by_id(id=self.name)
        # ✅ Best Practice: Initialize self.states as a dictionary
        if self.state_domain:
            self.state: dict = self.decode_state(self.state_domain.state)
        else:
            self.state = None

    # 🧠 ML Signal: Iterating over state_domains to populate self.states
    # 🧠 ML Signal: Usage of self.states.get to retrieve state by entity_id

    def persist_state(self):
        state_str = self.encode_state(self.state)
        # 🧠 ML Signal: String formatting pattern for domain_id
        if not self.state_domain:
            self.state_domain = self.state_schema(
                id=self.name, entity_id=self.name, state_name=self.name
            )
        # 🧠 ML Signal: Usage of self.state_schema.get_by_id to retrieve state domain
        self.state_domain.state = state_str
        self.state_session.add(self.state_domain)
        # 🧠 ML Signal: Encoding state using self.encode_state
        self.state_session.commit()


# 🧠 ML Signal: Creating a new state domain if it doesn't exist
# ⚠️ SAST Risk (Low): Committing to the database without exception handling
# ✅ Best Practice: Use of __all__ to define public API of the module
# 🧠 ML Signal: Setting state on state_domain
# 🧠 ML Signal: Adding state_domain to session
class EntityStateService(StatefulService):
    """
    StatefulService which saving one state one entity
    """

    def __init__(self, entity_ids) -> None:
        super().__init__()
        self.entity_ids = entity_ids
        state_domains: List[StateMixin] = self.state_schema.query_data(
            filters=[self.state_schema.state_name == self.name],
            entity_ids=self.entity_ids,
            return_type="domain",
        )

        #: entity_id:state
        self.states: dict = {}
        if state_domains:
            for state in state_domains:
                self.states[state.entity_id] = self.decode_state(state.state)

    def persist_state(self, entity_id):
        state = self.states.get(entity_id)
        if state:
            domain_id = f"{self.name}_{entity_id}"
            state_domain = self.state_schema.get_by_id(domain_id)
            state_str = self.encode_state(state)
            if not state_domain:
                state_domain = self.state_schema(
                    id=domain_id, entity_id=entity_id, state_name=self.name
                )
            state_domain.state = state_str
            self.state_session.add(state_domain)
            self.state_session.commit()


# the __all__ is generated
__all__ = ["StatefulService", "OneStateService", "EntityStateService"]
