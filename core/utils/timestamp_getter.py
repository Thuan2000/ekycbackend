import os

class TimestampGetter():
    def nas_timestamp(self, filepath):
        filename = os.path.basename(filepath)
        try:
            timestamp = filename.split('.')[0].split('-')[4]
        except IndexError:
            timestamp = filename.split('.')[0].split('-')[-1]
        return int(timestamp)
