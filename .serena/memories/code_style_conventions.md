# Code Style and Conventions

## General Style
- **Python Version**: Python 3.x
- **Import Style**: Explicit imports in `__init__.py` with `__all__` list
- **Package Structure**: Clear module separation by functionality

## Class and Function Design
- **Dataclasses**: Extensive use of `@dataclass` decorator for data structures
- **Type Hints**: Comprehensive type annotations using `typing` module
  - `Dict[str, Union[int, List[int]]]` for complex types
  - `Optional[Type]` for nullable parameters
  - `Tuple[str, str]` for structured returns
- **Enums**: Use `Enum` for constants (e.g., `GranularityLevel`)

## Documentation
- **Docstrings**: Triple-quoted strings for class and function documentation
- **Class Documentation**: Brief description of purpose
- **Method Documentation**: Describes purpose and parameters where complex

## Naming Conventions
- **Classes**: PascalCase (e.g., `BenchmarkPipeline`, `ModelMetadata`)
- **Functions/Methods**: snake_case (e.g., `load_and_prepare_data`, `run_single_model_experiment`)
- **Variables**: snake_case (e.g., `data_config`, `model_registry`)
- **Constants**: UPPER_CASE for enum values (e.g., `SKU`, `PRODUCT`)
- **Private Methods**: Leading underscore (e.g., `_make_json_serializable`)

## Code Organization
- **Method Grouping**: Related methods grouped together in classes
- **Error Handling**: Logging-based error reporting with structured messages
- **Configuration**: Separate config classes (`DataConfig`, `TrainingConfig`)
- **Factory Pattern**: Used for extensible model creation

## Import Organization
- Standard library imports first
- Third-party imports second  
- Local package imports last
- Clear `__all__` exports in `__init__.py`

## Logging
- Uses Python `logging` module
- Structured log messages with appropriate levels (INFO, WARNING, ERROR)
- Logger instances per module: `logger = logging.getLogger(__name__)`

## Data Handling
- **Memory Efficiency**: Prefer Polars LazyFrames over eager computation
- **Type Safety**: Strong typing for data structures
- **Immutability**: Dataclasses with default factories for mutable defaults