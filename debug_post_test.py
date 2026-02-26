import requests, tempfile, os
url='http://127.0.0.1:5000/api/debug_upload'
# Create temp files
face_path = os.path.join(tempfile.gettempdir(),'test_face.jpg')
voice_path = os.path.join(tempfile.gettempdir(),'test_voice.webm')
with open(face_path,'wb') as f: f.write(b'FAKEFACE')
with open(voice_path,'wb') as f: f.write(b'FAKEVOICE')
files={'face_image': open(face_path,'rb'), 'voice_audio': open(voice_path,'rb')}
data={'action':'IN','manual_date':'2026-02-06','manual_time':'22:00','pin':'1111'}
resp = requests.post(url, files=files, data=data)
print('STATUS', resp.status_code)
print(resp.text)
