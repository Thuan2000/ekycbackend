import requests
from requests import exceptions
from functools import wraps
import json
import os
import time
from utils.utils import recursive_retry
import pickle
import cv2
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class RequestError:
    Disconnect = 'RequestDisconnect'


def catch_disconnecting(func):

    def func_wrapper(*args, **kwargs):
        result = RequestError.Disconnect
        while result == RequestError.Disconnect:
            try:
                result = func(*args, **kwargs)
            except exceptions.ConnectionError:
                print('Caught disconnecting')
                time.sleep(5)
        return result

    return func_wrapper


class HTTPRequest():

    def __init__(self, url):
        self.url = url

    def post_list(self, key, image_list):
        # payload = {key : json.dumps(data)}
        data = [(key, i) for i in image_list]
        try:
            response = requests.post(self.url, data=data)
            return response
        except:
            return None


class TrackerHTTPRequest():

    def __init__(self, url, **kwargs):
        self.url = url
        self.path_prefix = kwargs.get('path_prefix', '')

    @catch_disconnecting
    def post_list(self, key, image_list):
        data = []
        for image_id in image_list:
            server_relative_path = os.path.join(self.path_prefix, image_id)
            data.append((key, server_relative_path))
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data:
                    return json_data['data']
        except Exception as e:
            print(e)
        return None


class VinaDataHTTPRequest():
    def __init__(self, authen_info, core_url, rec_url):
        self.core_url = core_url
        self.rec_url = rec_url
        self.username, self.password = authen_info
        self.token_url = url_join(self.core_url, '/accounts/authenticate')
        self.authenticate()

    def authenticate(self):
        print('Connecting to', self.token_url)
        authen_json = {'username': self.username,
                       'password': self.password}
        while True:
            try:
                response = requests.post(self.token_url, json=authen_json)
                if response.ok:
                    self.auth_token = response.json()['id_token']
                    self.auth_headers = {'Authorization': '%s %s' % ('Bearer', self.auth_token)}
                    break
                else:
                    print('Athentication status_code:', response.status_code)
                    continue
            except:
                print('Sleep 5s and try to re-authenticate')
                time.sleep(5)
                continue

    def get_playlist(self, camera_uuid, from_time, to_time):
        if from_time > 0 and to_time > from_time:
            url = url_join(self.rec_url, 'segment/create-playlist')
            payload = {'camUuid': camera_uuid,
                       'from': int(from_time),
                       'to': int(to_time)}
            response = self.get(url, params=payload)
            # no segment error record.playlist.error.no-segment-found
            if response.ok:
                playlist_url = response.json().get('playlistUrl', None)
                # print(playlist_url)
                file_urls = self.get_file_urls(playlist_url)
                return file_urls
        return []

    def get_file_urls(self, playlist_url):
        file_urls = []
        if playlist_url is not None:
            response = requests.get(playlist_url)
            if response.ok:
                playlist_lines = response.text.splitlines()
                for line in playlist_lines:
                    if 'http' in line:
                        file_urls.append(line)
        # print(file_urls)
        return file_urls

    def get_file(self, file_url):
        while True:
            try:
                response = requests.get(file_url, timeout=1000)
                if response.ok:
                    raw_file = response.content
                    return raw_file
                elif response.status_code == 401:
                    self.authenticate()
                    print('[INFO] Reauthenticating')
                    continue
                elif response.status_code != 200:
                    print('Trying: %s\ngot code: %s\n[INFO] Reauthenticating' % (file_url, response.status_code))
                    time.sleep(2)
                    self.authenticate()
                    time.sleep(2)
                # else:
                    continue
            except Exception as e:
                print('Got exception', e)
                time.sleep(5)
                continue
        return None

    def get(self, url, **kwargs):
        params = kwargs.get('params', {})
        while True:
            try:
                response = requests.get(url, headers=self.auth_headers,
                                        params=params, timeout=100)
                if response.status_code == 401:
                    self.authenticate()
                    continue
                return response
            except Exception as e:
                print('Got exception', e)
                time.sleep(5)
                continue
        return response


def url_join(*args):
    urls = [url.strip('/') for url in args]
    return '/'.join(urls)


class ImageTrackerHTTPRequest():

    def __init__(self, url='https://upload-web-service.eyeq.tech/upload'):
        self.url = url

    def post_list(self, upload_dir, file_name, file):
        data = [('uploadDir', upload_dir)]
        try:
            _, file = cv2.imencode('.jpg', file)
            response = requests.post(self.url, data=data, \
                files={'file': (file_name, file, 'image/jpeg')}, verify=False)
            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data:
                    return json_data['data']
        except:
            return None
