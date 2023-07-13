import aiofiles
import json
import os
import asyncio


async def read_json(file_path):
    async with aiofiles.open(file_path, mode="r") as file:
        contents = await file.read()
        data = json.loads(contents)
        return data


async def remove_file_async(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} removed.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")


async def remove_files_async(file_paths):
    tasks = []
    for file_path in file_paths:
        tasks.append(remove_file_async(file_path))
    await asyncio.gather(*tasks)
