import os
import time
import swagger_client
from swagger_client.rest import ApiException
# str | Account api key, to be used in every api call
swagger_client.configuration.api_key['apikey'] = os.getenv('MXM_API_KEY')

# print available APIs
# print(dir(swagger_client))

# create an instance of the API class
lyrics_api = swagger_client.LyricsApi()

# see some available methods
# print(dir(lyrics_api))

count = 0
while True:
    time.sleep(.1)
    print(count)
    count+=1
    try:
        res = lyrics_api.track_lyrics_get_get(1313157)
        # print(res)
        print(res)
        exit()
    except ApiException as api_ex:
        print(api_ex)
        print(count)
        exit()
    except Exception as ex:
        print(ex)
        print(count)
        exit()