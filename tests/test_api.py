import os, sys, tempfile, base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"backend"))
os.environ['DATABASE_URL']='sqlite:///'+tempfile.mktemp(suffix='.db')
os.environ['ADMIN_KEY']='test-only-key-01234567890123456789'
os.environ['EMBEDDING_KEY']=base64.urlsafe_b64encode(b'x'*32).decode()
import numpy as np
import app as m
from fastapi.testclient import TestClient

def test_workflow(monkeypatch):
    monkeypatch.setattr(m.vision,'sequence',lambda frames,direction:np.array([1.,0.]))
    with TestClient(m.app) as client:
        assert client.get('/api/students').status_code==401
        client.headers['X-Admin-Key']=os.environ['ADMIN_KEY']
        def capture(purpose,lesson=None):
            r=client.post('/api/challenge',json={'purpose':purpose,'lesson_id':lesson})
            assert r.status_code==200
            token=r.json()['id'];m.challenges[token]['time']-=4
            return {'challenge_id':token,'frames':['test']*6}
        b=capture('enroll');b.update(name='Test Student',code='ST-1',group='A',consent=True)
        assert client.post('/api/students',json=b).status_code==200
        assert client.post('/api/students',json=b).status_code==400
        lid=client.post('/api/lessons',json={'title':'Algorithms','group':'A'}).json()['id']
        for _ in range(2):
            b=capture('verify',lid);b['lesson_id']=lid
            assert client.post('/api/verify',json=b).status_code==200
        assert len(client.get('/api/attendance/'+str(lid)).json())==1
        other=client.post('/api/lessons',json={'title':'Other','group':'B'}).json()['id']
        b=capture('verify',other);b['lesson_id']=other
        assert client.post('/api/verify',json=b).status_code==422
        assert client.post('/api/manual',json={'student_id':1,'lesson_id':other,'reason':'Wrong group'}).status_code==400
        assert client.post('/api/lessons/'+str(lid)+'/close').status_code==200
        assert client.post('/api/challenge',json={'purpose':'verify','lesson_id':lid}).status_code==409
        assert client.delete('/api/students/1').status_code==200
        assert client.get('/api/students').json()==[]
        assert client.get('/api/attendance/'+str(lid)).json()==[]
