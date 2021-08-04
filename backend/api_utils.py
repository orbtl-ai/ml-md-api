import os

def clean_temporary_files(CHIP_IMAGE_PATH, FINAL_OUTPUT_PATH) -> None:
    '''
    A simple function to clean the API's temporary file storage locations. This is run both before and after each API run to 
    ensure no wires get crossed...
    '''
    for root, dirs, files in os.walk(CHIP_IMAGE_PATH):
      for file in files:
        os.remove(os.path.join(root, file))

    for root, dirs, files in os.walk(FINAL_OUTPUT_PATH):
      for file in files:
        os.remove(os.path.join(root, file))

def security_check(list_of_file_uploads, list_of_approved_content_types) -> list:
    '''
    A simple function designed to screen user uploads to check if they are a supported image type.
    This function could be modified in the future to also check other things, such as file size.
    '''
    screened_uploads = []
    for fi in list_of_file_uploads:
      if fi.content_type not in list_of_approved_content_types:
        print(f"File upload {fi.filename} will not be processed because content is not an approved file type ({list_of_approved_content_types})")
      else:
        screened_uploads.append(fi)
        
    return screened_uploads