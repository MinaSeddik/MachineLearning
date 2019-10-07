import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return

    duration = time.time() - start_time
    if duration == 0:
        return

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download(url, download_dir):
    # extract the file name from the given url
    file_path = urlparse(url)
    file_name = os.path.basename(file_path.path)

    # full path of the file
    save_path = os.path.join(download_dir, file_name)

    # Check if the file already exists, otherwise we need to download it now.
    if not os.path.exists(save_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

    try:

        # ssl._create_default_https_context = ssl._create_unverified_context
        # file_path, http_msg = urllib.request.urlretrieve(url=url,
        #                                                  filename=save_path,
        #                                                  reporthook=_reporthook)

        path = Path(save_path)
        with path.open('wb') as f:
            f.write(requests.get(url, allow_redirects=True, timeout=10, verify=False).content)

    except IOError as e:
        logger.error('Failed to download file from url: %s with IOError: ', url, str(e))
    except Exception as e:
        logger.error('Failed to download file from url: %s with Exception: ', url, str(e))

    return save_path
