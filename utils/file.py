import aiofiles
import json


async def read_json(file_path):
    async with aiofiles.open(file_path, mode="r") as file:
        contents = await file.read()
        data = json.loads(contents)
        return data
