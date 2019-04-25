from google_images_download import google_images_download

parameters = {'keywords': 'Teddybears, Grizzly bears, Black bears',
             'limit': 150,
             'print_urls': True,
             'chromedriver': 'chromedriver.exe',
             'output_directory': 'data'
              }

response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download(parameters)
