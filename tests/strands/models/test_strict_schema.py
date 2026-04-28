from strands.models._strict_schema import ensure_strict_json_schema


def test_ensure_strict_json_schema_basic():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    strict_schema = ensure_strict_json_schema(schema)

    assert strict_schema["additionalProperties"] is False
    assert strict_schema["properties"]["x"] == {"type": "string"}
    # Original should be untouched
    assert "additionalProperties" not in schema


def test_ensure_strict_json_schema_nested():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
            }
        },
    }
    strict_schema = ensure_strict_json_schema(schema)

    assert strict_schema["additionalProperties"] is False
    assert strict_schema["properties"]["outer"]["additionalProperties"] is False
    assert strict_schema["properties"]["outer"]["properties"]["inner"] == {"type": "integer"}


def test_ensure_strict_json_schema_with_defs():
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/MyItem"}},
        "$defs": {
            "MyItem": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
    }
    strict_schema = ensure_strict_json_schema(schema)

    assert strict_schema["additionalProperties"] is False
    assert strict_schema["$defs"]["MyItem"]["additionalProperties"] is False


def test_ensure_strict_json_schema_with_ref_inline():
    # When a $ref is combined with other keys, the reference should be inlined.
    # Note: ensure_strict_json_schema resolves refs from the root schema.
    schema = {
        "type": "object",
        "properties": {
            "item": {
                "$ref": "#/$defs/MyItem",
                "description": "An item",
            }
        },
        "$defs": {
            "MyItem": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
    }
    strict_schema = ensure_strict_json_schema(schema)

    assert strict_schema["additionalProperties"] is False
    # The reference should have been inlined, retaining 'description', and gaining additionalProperties
    item_prop = strict_schema["properties"]["item"]
    assert "$ref" not in item_prop
    assert item_prop["type"] == "object"
    assert item_prop["description"] == "An item"
    assert item_prop["additionalProperties"] is False
    assert item_prop["properties"]["name"] == {"type": "string"}


def test_ensure_strict_json_schema_arrays_and_unions():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
            "union": {
                "anyOf": [
                    {"type": "object", "properties": {"b": {"type": "string"}}},
                    {"type": "object", "properties": {"c": {"type": "string"}}},
                ]
            },
            "intersection": {
                "allOf": [
                    {"type": "object", "properties": {"d": {"type": "string"}}},
                ]
            },
        },
    }
    strict_schema = ensure_strict_json_schema(schema)

    assert strict_schema["additionalProperties"] is False
    assert strict_schema["properties"]["items"]["items"]["additionalProperties"] is False
    assert strict_schema["properties"]["union"]["anyOf"][0]["additionalProperties"] is False
    assert strict_schema["properties"]["union"]["anyOf"][1]["additionalProperties"] is False
    assert strict_schema["properties"]["intersection"]["allOf"][0]["additionalProperties"] is False


def test_ensure_strict_json_schema_require_all_properties():
    schema = {
        "type": "object",
        "properties": {
            "required_field": {"type": "string"},
            "optional_field": {"type": "string"},
        },
        "required": ["required_field"],
    }

    # Test without require_all_properties
    strict_schema = ensure_strict_json_schema(schema)
    assert strict_schema["required"] == ["required_field"]

    # Test with require_all_properties
    strict_req = ensure_strict_json_schema(schema, require_all_properties=True)
    # The order of keys is typically preserved from the dict iteration
    assert set(strict_req["required"]) == {"required_field", "optional_field"}
