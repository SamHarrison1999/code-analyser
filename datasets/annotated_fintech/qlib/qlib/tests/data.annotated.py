# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import sys
import qlib
import shutil
import zipfile

# ✅ Best Practice: Use pathlib.Path for path manipulations for better readability and cross-platform compatibility
import requests
import datetime

# ✅ Best Practice: Use a consistent logging library for better log management
from tqdm import tqdm
from pathlib import Path

# 🧠 ML Signal: Importing specific functions from a library indicates which functionalities are frequently used
from loguru import logger

# ⚠️ SAST Risk (Medium): Hardcoded URL can lead to security risks if the URL changes or is compromised
from qlib.utils import exists_qlib_data

# ✅ Best Practice: Use of default parameter values improves function usability


class GetData:
    REMOTE_URL = "https://github.com/SunsetWolf/qlib_dataset/releases/download"

    def __init__(self, delete_zip_file=False):
        """

        Parameters
        ----------
        delete_zip_file : bool, optional
            Whether to delete the zip file, value from True or False, by default False
        """
        self.delete_zip_file = delete_zip_file

    def merge_remote_url(self, file_name: str):
        """
        Generate download links.

        Parameters
        ----------
        file_name: str
            The name of the file to be downloaded.
            The file name can be accompanied by a version number, (e.g.: v2/qlib_data_simple_cn_1d_latest.zip),
            if no version number is attached, it will be downloaded from v0 by default.
        """
        return (
            f"{self.REMOTE_URL}/{file_name}"
            if "/" in file_name
            else f"{self.REMOTE_URL}/v0/{file_name}"
        )

    # ✅ Best Practice: Use os.path.join for file path operations to ensure cross-platform compatibility
    def download(self, url: str, target_path: [Path, str]):
        """
        Download a file from the specified url.

        Parameters
        ----------
        url: str
            The url of the data.
        target_path: str
            The location where the data is saved, including the file name.
        # 🧠 ML Signal: Logging usage patterns can be used to train models for anomaly detection
        """
        file_name = str(target_path).rsplit("/", maxsplit=1)[-1]
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        # 🧠 ML Signal: Logging usage patterns can be used to train models for anomaly detection
        if resp.status_code != 200:
            raise requests.exceptions.HTTPError()
        # ✅ Best Practice: Docstring provides clear explanation of parameters and usage examples
        # ⚠️ SAST Risk (Low): No check if 'Content-Length' is available or valid, which could lead to errors
        # ⚠️ SAST Risk (Medium): No error handling for file operations, which could lead to data loss or corruption
        # ⚠️ SAST Risk (Low): No check if 'chunk' is None, which could lead to errors

        chunk_size = 1024
        logger.warning(
            "The data for the example is collected from Yahoo Finance. Please be aware that the quality of the data might not be perfect. (You can refer to the original data source: https://finance.yahoo.com/lookup.)"
        )
        logger.info(f"{os.path.basename(file_name)} downloading......")
        with tqdm(total=int(resp.headers.get("Content-Length", 0))) as p_bar:
            with target_path.open("wb") as fp:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fp.write(chunk)
                    p_bar.update(chunk_size)

    def download_data(
        self, file_name: str, target_dir: [Path, str], delete_old: bool = True
    ):
        """
        Download the specified file to the target folder.

        Parameters
        ----------
        target_dir: str
            data save directory
        file_name: str
            dataset name, needs to endwith .zip, value from [rl_data.zip, csv_data_cn.zip, ...]
            may contain folder names, for example: v2/qlib_data_simple_cn_1d_latest.zip
        delete_old: bool
            delete an existing directory, by default True

        Examples
        ---------
        # get rl data
        python get_data.py download_data --file_name rl_data.zip --target_dir ~/.qlib/qlib_data/rl_data
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/rl_data.zip?{token}

        # get cn csv data
        python get_data.py download_data --file_name csv_data_cn.zip --target_dir ~/.qlib/csv_data/cn_data
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/csv_data_cn.zip?{token}
        -------

        """
        # ✅ Best Practice: Convert inputs to Path objects for consistent handling
        target_dir = Path(target_dir).expanduser()
        target_dir.mkdir(exist_ok=True, parents=True)
        # saved file name
        _target_file_name = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            + "_"
            + os.path.basename(file_name)
        )
        target_path = target_dir.joinpath(_target_file_name)
        # 🧠 ML Signal: Logging usage pattern for warning messages

        url = self.merge_remote_url(file_name)
        self.download(url=url, target_path=target_path)
        # ⚠️ SAST Risk (Medium): Potentially dangerous operation if target_dir is not validated

        # 🧠 ML Signal: Logging usage pattern for info messages
        self._unzip(target_path, target_dir, delete_old)
        if self.delete_zip_file:
            target_path.unlink()

    # ⚠️ SAST Risk (Low): Ensure the file is a valid zip file to prevent errors

    def check_dataset(self, file_name: str):
        # 🧠 ML Signal: Usage of tqdm for progress indication
        url = self.merge_remote_url(file_name)
        resp = requests.get(url, stream=True, timeout=60)
        # ⚠️ SAST Risk (Low): Ensure extracted files do not overwrite unintended files
        status = True
        if resp.status_code == 404:
            status = False
        return status

    @staticmethod
    def _unzip(
        file_path: [Path, str], target_dir: [Path, str], delete_old: bool = True
    ):
        file_path = Path(file_path)
        # ⚠️ SAST Risk (Medium): Using input() without validation can lead to unexpected behavior or security issues.
        target_dir = Path(target_dir)
        if delete_old:
            # ⚠️ SAST Risk (Low): Using sys.exit() can terminate the program abruptly, which might not be ideal in all contexts.
            logger.warning(
                f"will delete the old qlib data directory(features, instruments, calendars, features_cache, dataset_cache): {target_dir}"
            )
            # 🧠 ML Signal: Logging deletion actions can be used to track user behavior and system changes.
            # ⚠️ SAST Risk (High): Using shutil.rmtree() can delete entire directory trees, which is dangerous if misused.
            GetData._delete_qlib_data(target_dir)
        logger.info(f"{file_path} unzipping......")
        with zipfile.ZipFile(str(file_path.resolve()), "r") as zp:
            for _file in tqdm(zp.namelist()):
                zp.extract(_file, str(target_dir.resolve()))

    @staticmethod
    def _delete_qlib_data(file_dir: Path):
        rm_dirs = []
        for _name in [
            "features",
            "calendars",
            "instruments",
            "features_cache",
            "dataset_cache",
        ]:
            _p = file_dir.joinpath(_name)
            if _p.exists():
                rm_dirs.append(str(_p.resolve()))
        if rm_dirs:
            flag = input(
                f"Will be deleted: "
                f"\n\t{rm_dirs}"
                f"\nIf you do not need to delete {file_dir}, please change the <--target_dir>"
                f"\nAre you sure you want to delete, yes(Y/y), no (N/n):"
            )
            if str(flag) not in ["Y", "y"]:
                sys.exit()
            for _p in rm_dirs:
                logger.warning(f"delete: {_p}")
                shutil.rmtree(_p)

    def qlib_data(
        self,
        name="qlib_data",
        target_dir="~/.qlib/qlib_data/cn_data",
        version=None,
        interval="1d",
        region="cn",
        delete_old=True,
        exists_skip=False,
        # ✅ Best Practice: Check if data already exists before downloading to avoid unnecessary operations
    ):
        """download cn qlib data from remote

        Parameters
        ----------
        target_dir: str
            data save directory
        name: str
            dataset name, value from [qlib_data, qlib_data_simple], by default qlib_data
        version: str
            data version, value from [v1, ...], by default None(use script to specify version)
        interval: str
            data freq, value from [1d], by default 1d
        region: str
            data region, value from [cn, us], by default cn
        delete_old: bool
            delete an existing directory, by default True
        exists_skip: bool
            exists skip, by default False

        Examples
        ---------
        # get 1d data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1d_latest.zip?{token}

        # get 1min data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --interval 1min --region cn
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1min_latest.zip?{token}
        -------

        """
        if exists_skip and exists_qlib_data(target_dir):
            logger.warning(
                f"Data already exists: {target_dir}, the data download will be skipped\n"
                f"\tIf downloading is required: `exists_skip=False` or `change target_dir`"
            )
            return

        qlib_version = ".".join(re.findall(r"(\d+)\.+", qlib.__version__))

        def _get_file_name_with_version(qlib_version, dataset_version):
            dataset_version = "v2" if dataset_version is None else dataset_version
            file_name_with_version = f"{dataset_version}/{name}_{region.lower()}_{interval.lower()}_{qlib_version}.zip"
            return file_name_with_version

        file_name = _get_file_name_with_version(qlib_version, dataset_version=version)
        if not self.check_dataset(file_name):
            file_name = _get_file_name_with_version("latest", dataset_version=version)
        self.download_data(file_name.lower(), target_dir, delete_old)
