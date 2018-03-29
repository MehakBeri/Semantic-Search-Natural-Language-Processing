import pysolr
import Constants
from Constants import Constants

Constant = Constants()
class SolrCommunicator:

	instance = None
	index = 0
	solr = None
	def __init__(self):
		SolrCommunicator.instance = self
		SolrCommunicator.index +=1
		SolrCommunicator.solr = SolrCommunicator.createSolrConnection()

	def __call__(self):
		return SolrCommunicator.instance

	@staticmethod
	def getInstance():
		if SolrCommunicator.instance is None:
			SolrCommunicator()
		return SolrCommunicator.instance

	@staticmethod
	def createSolrConnection():
		solr = pysolr.Solr(Constant.solrUrl + Constant.project, timeout=Constant.solrTimeOut)
		return solr

	@staticmethod
	def addDataToSolr(data):
		if SolrCommunicator.solr is not None:
			return SolrCommunicator.solr.add(data)
		else:
			print('solr connection is not created, cannot add data')
			return None

	@staticmethod
	def solrSearch(words, fl, rows, sort):
		if SolrCommunicator.solr is not None:
			return SolrCommunicator.solr.search(
			    words, **{
			    'fl': fl,
				'rows': rows,
				'sort': sort
			})
		else:
			print('solr connection is not created, cannot search data')
			return None
