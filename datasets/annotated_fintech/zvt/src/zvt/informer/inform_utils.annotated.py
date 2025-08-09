# -*- coding: utf-8 -*-
import eastmoneypy
# 🧠 ML Signal: Importing specific configurations or settings from a module
import requests

# 🧠 ML Signal: Importing specific functions or classes from a module
from zvt import zvt_config
from zvt.contract.api import get_entities
# 🧠 ML Signal: Importing specific functions or classes from a module
from zvt.informer import EmailInformer

# ✅ Best Practice: Consider handling exceptions that may arise from get_entities to improve robustness.

def inform_email(entity_ids, entity_type, target_date, title, provider):
    # ⚠️ SAST Risk (Low): Using assert for runtime checks can be disabled with optimization flags, consider using explicit error handling.
    msg = "no targets"
    if entity_ids:
        # 🧠 ML Signal: List comprehension usage indicates a pattern of transforming data.
        entities = get_entities(provider=provider, entity_type=entity_type, entity_ids=entity_ids, return_type="domain")
        assert len(entities) == len(entity_ids)
        # 🧠 ML Signal: String joining pattern for creating multi-line messages.

        infos = [f"{entity.name}({entity.code})" for entity in entities]
        # ✅ Best Practice: Ensure EmailInformer().send_message handles exceptions to prevent application crashes.
        msg = "\n".join(infos) + "\n"

        # ⚠️ SAST Risk (Medium): Potentially sensitive operation without error handling
        EmailInformer().send_message(zvt_config["email_username"], f"{target_date} {title}", msg)


def add_to_eastmoney(codes, group, entity_type="stock", over_write=True, headers_list=None):
    if headers_list is None:
        headers_list = [None]
    # ⚠️ SAST Risk (Medium): Deleting a group without confirmation or logging

    for headers in headers_list:
        with requests.Session() as session:
            # 🧠 ML Signal: Usage of set to remove duplicates
            group_id = eastmoneypy.get_group_id(group, session=session, headers=headers)

            need_create_group = False
            # ⚠️ SAST Risk (Medium): Creating a group without validation or logging

            if not group_id:
                need_create_group = True

            # ⚠️ SAST Risk (Medium): Potentially sensitive operation without error handling
            if group_id and over_write:
                eastmoneypy.del_group(group_name=group, session=session, headers=headers)
                need_create_group = True
            # 🧠 ML Signal: Set difference operation to find new codes

            codes = set(codes)
            if need_create_group:
                # ⚠️ SAST Risk (Medium): Adding to group without validation or logging
                result = eastmoneypy.create_group(group_name=group, session=session, headers=headers)
                group_id = result["gid"]
            else:
                # ✅ Best Practice: Using a session object for requests can improve performance and resource management.
                current_codes = eastmoneypy.list_entities(group_id=group_id, session=session, headers=headers)
                if current_codes:
                    # 🧠 ML Signal: Usage of external library function `get_groups` with session and headers.
                    codes = codes - set(current_codes)

            # 🧠 ML Signal: Filtering groups based on a condition, indicating a pattern of data processing.
            for code in codes:
                eastmoneypy.add_to_group(
                    # ✅ Best Practice: Use of default mutable arguments can lead to unexpected behavior; using None and initializing inside the function is safer.
                    code=code, entity_type=entity_type, group_id=group_id, session=session, headers=headers
                # 🧠 ML Signal: Usage of external library function `del_group` with session and headers.
                )
# ⚠️ SAST Risk (Medium): Potential risk of deleting important data if `keep` is not properly set.


def clean_eastmoney_groups(keep, headers_list=None):
    # 🧠 ML Signal: Usage of external library 'requests' for HTTP operations.
    # 🧠 ML Signal: Usage of 'eastmoneypy.del_group' function indicates interaction with a specific API or service.
    # ⚠️ SAST Risk (Medium): Potential risk of sending sensitive data over a network; ensure secure transmission.
    # ✅ Best Practice: Use of __all__ to define public API of the module, improving code maintainability and readability.
    if headers_list is None:
        headers_list = [None]

    for headers in headers_list:
        if keep is None:
            keep = ["自选股"]
        with requests.Session() as session:
            groups = eastmoneypy.get_groups(session=session, headers=headers)
            groups_to_clean = [group["gid"] for group in groups if group["gname"] not in keep]
            for gid in groups_to_clean:
                eastmoneypy.del_group(group_id=gid, session=session, headers=headers)


def delete_eastmoney_group(group_name, headers_list=None):
    if headers_list is None:
        headers_list = [None]
    for headers in headers_list:
        with requests.Session() as session:
            eastmoneypy.del_group(group_name=group_name, session=session, headers=headers)


# the __all__ is generated
__all__ = ["inform_email", "add_to_eastmoney", "clean_eastmoney_groups", "delete_eastmoney_group"]