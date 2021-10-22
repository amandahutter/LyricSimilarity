import os
import swagger_client
from swagger_client.rest import ApiException
# str | Account api key, to be used in every api call
swagger_client.configuration.api_key['apikey'] = os.getenv('MXM_API_KEY')

# create an instance of the API class
api_instance = swagger_client.TrackApi()

try:
    req = api_instance.track_get_get(6447080)
    print(dir(req))
    print(req)
except ApiException as api_ex:
    print(api_ex)
except Exception as ex:
    print(ex)