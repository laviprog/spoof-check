# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/laviprog/spoof-check/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                   |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/\_\_init\_\_.py                    |        0 |        0 |        0 |        0 |    100% |           |
| src/api/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| src/api/anti\_spoofing/\_\_init\_\_.py |        3 |        0 |        0 |        0 |    100% |           |
| src/api/anti\_spoofing/client.py       |       75 |       12 |       14 |        4 |     82% |30, 38, 41, 57-58, 103-104, 110, 115-118 |
| src/api/anti\_spoofing/schema.py       |       22 |        0 |        0 |        0 |    100% |           |
| src/api/base\_client.py                |       15 |        0 |        0 |        0 |    100% |           |
| src/config.py                          |       20 |        1 |        0 |        0 |     95% |        27 |
| src/core/\_\_init\_\_.py               |        0 |        0 |        0 |        0 |    100% |           |
| src/services/\_\_init\_\_.py           |        0 |        0 |        0 |        0 |    100% |           |
| src/services/audio.py                  |       48 |       35 |        4 |        0 |     25% |20-23, 26-28, 34-50, 59-64, 70-92 |
| src/utils.py                           |        3 |        0 |        0 |        0 |    100% |           |
| src/web/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| src/web/gradio\_app.py                 |       94 |       63 |       16 |        0 |     30% |18-20, 34-48, 51-73, 80-100, 132-146, 152-255, 267-276, 289 |
| **TOTAL**                              |  **280** |  **111** |   **34** |    **4** | **58%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/laviprog/spoof-check/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/laviprog/spoof-check/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/laviprog/spoof-check/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/laviprog/spoof-check/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flaviprog%2Fspoof-check%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/laviprog/spoof-check/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.