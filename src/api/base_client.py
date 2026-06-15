import httpx


class BaseClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}

    async def _post(self, endpoint: str, timeout: float = 60.0, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
            verify=False,
        ) as client:
            response = await client.post(endpoint, **kwargs)
            response.raise_for_status()
            return response

    async def _get(self, endpoint: str, timeout: float = 60.0, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
            verify=False,
        ) as client:
            response = await client.get(endpoint, **kwargs)
            response.raise_for_status()
            return response
