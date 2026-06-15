# DEVELOPMENT_GUIDELINES.md

## Coding Standards

Language:

Python 3.11+

---

## Style

* PEP8 compliant
* Type hints required
* Docstrings required
* Modular design

---

## Logging

Use centralized logging.

Never use print statements for production logic.

---

## Error Handling

Handle all recoverable exceptions.

Provide meaningful error messages.

---

## Testing

Required:

* Unit Tests
* Integration Tests
* End-to-End Tests

---

## Machine Learning Standards

Persist:

* Models
* Encoders
* Scalers
* Thresholds

using Joblib.

---

## Configuration

Never hardcode:

* Paths
* Thresholds
* Hyperparameters

Store in config files.

---

## Documentation

Every public function requires:

* Purpose
* Inputs
* Outputs
* Exceptions

---

## Performance

Prefer efficient algorithms.

Optimize for:

* Low latency
* High throughput
* Reproducibility
