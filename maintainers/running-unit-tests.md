# Running PyTest Unit Tests

This document provides instructions on how to run the PyTest unit tests for the MemMachine project. Running these tests is a critical step to verify that your changes have not introduced any regressions and that the codebase remains stable.

## Prerequisites

Before running the tests, ensure you have the following installed:

1. **Python:** Make sure you have Python 3.12 or higher installed.
2. **Poetry:** This project uses Poetry for dependency management. If you don't have it, you can install it by following the official [Poetry installation guide](https://python-poetry.org/docs/#installation).
3. **Project Dependencies:** Install all the required dependencies, including the ones for development, by running the following command in the project's root directory:

    ```bash
    poetry install --all-extras
    ```

4. **Additional Test Dependencies:** Install `pytest-asyncio` for running asynchronous tests and download the necessary NLTK data:

    ```bash
    pip install pytest-asyncio
    python3 -c "import nltk; nltk.download('punkt');nltk.download('punkt_tab'); nltk.download('stopwords')"
    ```

**A Note on Poetry:** This project uses Poetry to manage dependencies and ensure a consistent development environment. While you can install dependencies manually using `pip`, we strongly recommend using Poetry to avoid potential conflicts and to make sure you have the exact versions of the packages that are used in the project.

## Running the Tests

Once your environment is set up, you can run the unit tests using a variety of `pytest` commands. Here are some common examples that will be useful for your daily activities as a maintainer.

### Running All Tests

To run the entire test suite, ensure you are in the root of the MemMachine project, then run `pytest` with no arguments:

```bash
pytest
```

By default, this will discover and run all tests under the `tests/` directory.  
You can also explicitly specify the directory, which may make it clearer where tests are coming from:

```bash
pytest /test
```

### Verbose Output

For more detailed output, which can be helpful for debugging, use the `-v` flag:

```bash
pytest -v tests/
```

### Displaying Print Statements

By default, `pytest` captures output from `print()` statements. To display them in the console, which is useful for debugging, use the `-s` flag:

```bash
pytest -s tests/
```

### Running a Specific Test File

To run all the tests in a single file, provide the path to that file:

```bash
pytest tests/memmachine/server/test-rest-memories.py
```

### Running a Single Test Function

To run a specific test function within a file, use the `::` syntax:

```bash
pytest tests/memmachine/server/test-rest-memories.py::test_create_memory_sync
```

### Running Tests with a Keyword

You can also run tests that have a specific keyword in their name using the `-k` flag. This is useful for running a group of related tests.

```bash
pytest -k "create_memory"
```

These commands are essential for efficient testing and will also serve as a foundation for automating the unit test process in the future.

## Watching for Failures

Pay close attention to the output of the `pytest` command. If any tests fail, the output will provide detailed information about the failure, including the file, the function, and the line of code that caused the error.

Before submitting a pull request, all unit tests must pass. If you have a failing test, you should fix it before proceeding.
