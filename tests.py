import unittest
import json,requests
class TestMethods(unittest.TestCase):
	"""docstring for TestMethods"""
	def test(self):
		url= 'http://localhost:5004/api'
		data = json.dumps(['Congratulations ur awarded either å£500 of CD '])
		r=requests.post(url,data)
		result = r.json()
		self.assertEqual(result['results'],'spam')
if __name__ == '__main__':
	unittest.main()
