---
applyTo: '**/voice_live/**'
description: 'Azure Voice Live SDK conventions and known issues for this project'
---

# Azure Voice Live SDK Instructions

## Discriminator Type Fields — Always Set Explicitly

Azure Voice Live model classes use a `type` discriminator for polymorphic
serialization. **Not all classes auto-populate this field.**

Known inconsistency (SDK v1.1.0):

| Class                    | Auto-sets `type`? | Must pass explicitly? |
|--------------------------|-------------------|-----------------------|
| `AudioEchoCancellation`  | Yes               | No                    |
| `AudioNoiseReduction`    | **No**            | **Yes**               |

### Valid `type` literals for `AudioNoiseReduction`

- `"azure_deep_noise_suppression"` (recommended default)
- `"near_field"`
- `"far_field"`

Reference: <https://learn.microsoft.com/en-us/python/api/azure-ai-voicelive/azure.ai.voicelive.models.audionoisereduction>

### Verification Pattern

After constructing any Voice Live config model, verify serialization:

```python
obj = AudioNoiseReduction(type="azure_deep_noise_suppression")
assert "type" in obj.as_dict(), "Missing discriminator — service will reject this"
```

### What Happens If You Forget

`AudioNoiseReduction()` serializes to `{}`. The service returns:

```
Unable to extract tag using discriminator 'type'
```

This cascades to `max_config_attempts_exceeded` (5 retries) and the WebSocket
closes, killing the session.
