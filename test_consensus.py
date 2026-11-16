"""Testes sinteticos numericos do consenso; nao requer broker."""
import importlib.util
from pathlib import Path
import numpy as np
import unittest
import sys, types
try:
 import paho.mqtt.client
except ImportError:
 paho=types.ModuleType('paho'); mqtt=types.ModuleType('paho.mqtt');client=types.ModuleType('paho.mqtt.client')
 class Dummy:
  def __init__(self,*a,**kw): pass
 client.Client=Dummy; mqtt.client=client;paho.mqtt=mqtt
 sys.modules.update({'paho':paho,'paho.mqtt':mqtt,'paho.mqtt.client':client})
spec=importlib.util.spec_from_file_location('server',Path(__file__).parent/'server'/'fedkd_server.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
class ConsensusTests(unittest.TestCase):
 def create(self):
  return mod.FedKDServer('localhost',1883,2,5,2.0,log_dir='/tmp/fedkd-bloco3-unit',client_weights={'a':1,'b':3},sharpening=0.7)
 def test_weighted(self):
  s=self.create()
  s._logits={'a':np.array([[0,2,0,0]],dtype='f4'),'b':np.array([[0,0,2,0]],dtype='f4')}
  got=s._aggregate()[0]
  z=np.array([0,0.5,1.5,0],dtype=float)
  exp=np.exp(z/(2.0*0.7)-np.max(z/(2.0*0.7)));exp/=exp.sum()
  np.testing.assert_allclose(got,exp,rtol=1e-6,atol=1e-7)
 def test_identical_logits(self):
  s=self.create(); x=np.array([[2.,-1.,0.,3.]],dtype='f4')
  s._logits={'a':x,'b':x.copy()};out=s._aggregate()
  np.testing.assert_allclose(out.sum(axis=1),[1.0],atol=1e-7)
  self.assertTrue((out>0).all())
 def test_uniform(self):
  s=self.create();s._logits={'a':np.zeros((2,4),dtype='f4'),'b':np.zeros((2,4),dtype='f4')}
  np.testing.assert_allclose(s._aggregate(), np.full((2,4),0.25,dtype='f4'))
if __name__=='__main__':unittest.main()
