import os
import casbin
from config import CASBIN_MODEL_PATH, CASBIN_POLICY_PATH
from database import get_user_role

enforcer = casbin.Enforcer(CASBIN_MODEL_PATH, CASBIN_POLICY_PATH)

def check_permission(user_id: int, resource: str, action: str) -> bool:
    role = get_user_role(user_id)
    return enforcer.enforce(role, resource, action)

def check_permission_by_role(role: str, resource: str, action: str) -> bool:
    return enforcer.enforce(role, resource, action)