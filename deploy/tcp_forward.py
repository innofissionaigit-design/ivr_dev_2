"""Forward a public port to a local one, byte for byte.

RunPod only routes HTTP proxy traffic to the ports listed under "Expose HTTP
ports" in the pod template; on this pod that is 8888 (Jupyter's). Rather than
load a second copy of every speech model just to listen on 8888, this relays
TCP to the voice agent that is already running. Plain TCP relaying carries
HTTP and the WebSocket upgrade unchanged, so /ws/audio works through it.

    python deploy/tcp_forward.py 8888 8101
"""
import asyncio
import sys


async def _pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def main(listen: int, target: int) -> None:
    async def handle(cr, cw):
        try:
            tr, tw = await asyncio.open_connection("127.0.0.1", target)
        except OSError:
            cw.close()
            return
        await asyncio.gather(_pipe(cr, tw), _pipe(tr, cw))

    server = await asyncio.start_server(handle, "0.0.0.0", listen)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
