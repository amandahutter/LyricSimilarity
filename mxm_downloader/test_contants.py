# Need this to convert output copied from console into an object.
# We access api object responses via attribute, not dictionary key.
# TODO: We may be able to find a relevant object in the generated swagger library instead of this,
# and just construct the object that way.
from munch import munchify

EXAMPLE_LYRICS = """What can I see for ...?
What can I speak for ...?
What can I touch for... free?

Suffer to feel relief
Kiss to taste some blood
Sleep to stay awake
Die to feel alive

...

******* This Lyrics is NOT for Commercial use *******
(1409622193474)
"""

EXAMPLE_LYRICS_STRIPPED = """What can I see for ...?
What can I speak for ...?
What can I touch for... free?

Suffer to feel relief
Kiss to taste some blood
Sleep to stay awake
Die to feel alive

...

"""

# Convert the console output I copied into an object. 
LYRICS_RESPONSE = munchify({'action_requested': None,
 'can_edit': None,
 'explicit': 0.0,
 'html_tracking_url': None,
 'instrumental': None,
 'locked': None,
 'lyrics_body': 'What can I see for ...?\n'
                'What can I speak for ...?\n'
                'What can I touch for... free?\n'
                '\n'
                'Suffer to feel relief\n'
                'Kiss to taste some blood\n'
                'Sleep to stay awake\n'
                'Die to feel alive\n'
                '\n'
                '...\n'
                '\n'
                '******* This Lyrics is NOT for Commercial use *******\n'
                '(1409622193474)',
 'lyrics_copyright': 'Lyrics powered by www.musixmatch.com. This Lyrics is NOT '
                     'for Commercial use and only 30% of the lyrics are '
                     'returned.',
 'lyrics_id': 1371667.0,
 'lyrics_language': None,
 'lyrics_language_description': None,
 'pixel_tracking_url': 'https://tracking.musixmatch.com/t1.0/m_img/e_1/sn_0/l_1371667/su_0/rs_0/tr_3vUCAB8M2ObpwIfPXA4lor0RUnIWCJ9r2Twm8wyGbljOeFa6f1chEAAOvLBDgG_6qfGlCgrOWLFwIo6AgBC06nlJP6V0DAbZgYB2a9rvLMj6OKPpq6Onw6r8OPzZXiOpWxmreuDeHIUrYTnVP_J6Xb-TfOhPs3GROYFxOZgGmLPVQSXhkcnkxAX19cuR2fyWjysi_eUOqOfV9G7FnpZqzLKgMEeWJ5XmLc2eJ7bzh8x-iFjTcpxjNSXXkSxVGTJGTMOoLUU1t9C9GUxvJdAx1tHwuByudJa8JhQ97D4un_f33JRhrETdiYydjlpCNjwmESUhxcrJ6RYyA4fpf4GLVPBlt4a-hM82lpnneg4eRIOEnyZ2sP-sE5_IEVT7luB0kvG-eJfqHY-LgCrYziI04KxSXjEDAJIqwEx8-DYI3rp-pp6gr7xdczSj0Lvu39qliAPp/',
 'publisher_list': None,
 'restricted': None,
 'script_tracking_url': 'https://tracking.musixmatch.com/t1.0/m_js/e_1/sn_0/l_1371667/su_0/rs_0/tr_3vUCACFJPYiaFpc3Xmq2jKtDiwitEKFHAhm7LOJd165CoL_dvQKdIwgXoZulps0KpiH01CRlcdhNXDkGG9_pSy9sUxIJUSGJKQcAyRQ704O_nhkHpImv3nHvNUxPEGJrbyMoBa7dsfHNTyqXhoBNp5ChYzKGR0cT2_f7PAdFrDb2c322NwFJ14oVpq2fz1VbXCf0QAaQRtfHOis4qyi8Z5sUKvOOUH3GQvENkuqUZE_VFbnisgTDbdSEjxfR77snFad591AfYYthT-ZIq3UWdBNr-Fx8CnRx02MAVrqse_3WCKOnZKsT3pMoqG16VUu_ti441uQoqbaKHcL3i6_WEqauOvneY8WdkiFw-m-NY5795uNgunZOzyV2tmxKwbmOg95zz_3nIjPKUDVD6T2amtZaM3RnKvAiNW7ev5q_6ivWocD6fAyL6s3BwdHZyArx0NxY/',
 'updated_time': '2021-07-08T13:29:05Z',
 'verified': None,
 'writer_list': None})

MXM_ID = 1313157

LYRICS_TUPLE = (
    MXM_ID,
    1371667,
    EXAMPLE_LYRICS_STRIPPED,
    0,
    None,
    None
)