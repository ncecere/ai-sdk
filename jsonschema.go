package ai

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// JSONSchemaFromType builds a simple JSON Schema document for the
// Go type of example and returns it as a raw JSON byte slice.
//
// This helper is intentionally conservative and is meant for
// straightforward use cases (basic structs, slices, maps, and
// primitive types). It does not aim to support the full JSON
// Schema specification.
//
// Rules and limitations:
//   - Structs become objects with properties derived from exported
//     fields. Field names follow the `json` struct tag when present
//     (ignoring `,omitempty`), otherwise the field name is used.
//   - Pointer fields, slices, maps, and structs are treated as
//     optional; other fields are considered required.
//   - Maps become `{"type":"object","additionalProperties":...}`
//     where the value schema is derived from the map element type.
//   - Unsupported or unknown kinds default to `{ "type": "string" }`.
func JSONSchemaFromType(example any) ([]byte, error) {
	t := reflect.TypeOf(example)
	if t == nil {
		return nil, fmt.Errorf("jsonschema: nil example type")
	}
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	schema := schemaForType(t)
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func schemaForType(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Slice, reflect.Array:
		return map[string]any{
			"type":  "array",
			"items": schemaForType(t.Elem()),
		}
	case reflect.Map:
		return map[string]any{
			"type":                 "object",
			"additionalProperties": schemaForType(t.Elem()),
		}
	case reflect.Struct:
		props := make(map[string]any)
		var required []string
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name, omit := jsonFieldName(f)
			if name == "" {
				continue
			}
			props[name] = schemaForType(indirectType(f.Type))
			if !omit && !isOptionalKind(f.Type.Kind()) {
				required = append(required, name)
			}
		}
		m := map[string]any{
			"type":       "object",
			"properties": props,
		}
		if len(required) > 0 {
			m["required"] = required
		}
		return m
	default:
		// Fallback for unsupported kinds.
		return map[string]any{"type": "string"}
	}
}

func jsonFieldName(f reflect.StructField) (name string, omit bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false
	}
	if tag == "" {
		return f.Name, false
	}
	// Take token before first comma.
	for i, ch := range tag {
		if ch == ',' {
			if i == 0 {
				return f.Name, true
			}
			return tag[:i], true
		}
	}
	return tag, false
}

func indirectType(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return t
}

func isOptionalKind(k reflect.Kind) bool {
	switch k {
	case reflect.Ptr, reflect.Slice, reflect.Map:
		return true
	default:
		return false
	}
}
