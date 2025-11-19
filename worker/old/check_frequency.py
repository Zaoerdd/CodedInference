import asyncio
import traceback

import asyncssh


async def run_command(host, command, known_hosts=None):
    async with asyncssh.connect(host, username='pi', password='88888888',
                                known_hosts=known_hosts) as conn:
        result = await conn.run(command)
        return host, result.stdout, result.stderr


async def main(hosts, command):
    tasks = [run_command(host, command) for host in hosts]
    results = await asyncio.gather(*tasks)
    for host, stdout, stderr in results:
        print(f"Host: {host}")
        print(stdout, end='')
        if stderr:
            print(f"STDERR:\n{stderr}")


if __name__ == "__main__":
    hosts = ['192.168.1.110',
             '192.168.1.112',
             '192.168.1.113',
             '192.168.1.114',
             '192.168.1.115',
             '192.168.1.116',
             '192.168.1.201',
             '192.168.1.203',
             '192.168.1.206',
             '192.168.1.207']  # 替换为你的远程主机IP或域名
    command = 'cpupower frequency-info | grep current | grep Hz'  # change the command if necessary
    # command = 'sudo cpupower -c all frequency-set -u 600MHz'
    # hosts = ['192.168.1.110']
    # command = 'sudo halt'

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(hosts, command))
    except Exception:
        traceback.print_exc()
    finally:
        loop.close()
