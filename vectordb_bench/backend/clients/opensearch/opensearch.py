import logging
import time
from contextlib import contextmanager
from typing import Iterable
from ..api import VectorDB
from .config import OpensearchIndexConfig
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch

for logger in ("opensearchpy"):
    logging.getLogger(logger).setLevel(logging.WARNING)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Opensearch(VectorDB):
    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: OpensearchIndexConfig,
            indice: str = "vdb_bench_indice",  # must be lowercase
            id_col_name: str = "id",
            vector_col_name: str = "vector",
            drop_old: bool = False,
            **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.case_config.dimensions = dim
        self.indice = indice
        self.id_col_name = id_col_name
        self.vector_col_name = vector_col_name

        from opensearchpy import OpenSearch

        log.info(f"Db config is : {self.db_config['url']}")

        client = OpenSearch(
            hosts=[{'host': self.db_config['url'], 'port': 443}],
            http_auth=self.db_config['basic_auth'],
            http_compress=True,  # enables gzip compression for request bodies
            use_ssl=True,
            verify_certs=True
        )

        if drop_old:
            log.info(f"Opensearch client drop_old indices: {self.indice}")
            is_existed_res = client.indices.exists(self.indice)
            if is_existed_res:
                client.indices.delete(index=self.indice)
            self._create_indice(client)

    @contextmanager
    def init(self) -> None:
        """connect to elasticsearch"""
        from opensearchpy import OpenSearch
        self.client: OpenSearch = OpenSearch(
            hosts=[{'host': self.db_config['url'], 'port': 443}],
            http_auth=self.db_config['basic_auth'],
            http_compress=True,  # enables gzip compression for request bodies
            use_ssl=True,
            verify_certs=True
        )

        yield
        # self.client.transport.close()
        self.client = None
        del (self.client)

    def _create_indice(self, client: OpenSearch) -> None:
        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "refresh_interval": "60s",
                    "number_of_shards": self.case_config.primaries,
                    "number_of_replicas": self.case_config.replicas,
                    "knn.algo_param.ef_search": self.case_config.efSearch
                }
            },
            "mappings": {
                "_source": {"excludes": [self.vector_col_name]},
                "properties": {
                    self.id_col_name: {"type": "integer", "store": True},
                    self.vector_col_name: {
                        **self.case_config.index_param(),
                    }
                }
            }
        }

        try:
            client.indices.create(index=self.indice, body=body)
        except Exception as e:
            log.warning(f"Failed to create indice: {self.indice} error: {str(e)}")
            raise e from None

    def insert_embeddings(
            self,
            embeddings: Iterable[list[float]],
            metadata: list[int],
            **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the opensearch."""
        assert self.client is not None, "should self.init() first"
        log.info("Doing Indexing............")

        def gen():
            for i in range(len(embeddings)):
                yield {"_index": self.indice, self.vector_col_name: embeddings[i],
                       self.id_col_name: metadata[i]}

        try:
            # bulk_insert_res = bulk(self.client, insert_data)
            (_, errors) = bulk(self.client, actions=gen(), stats_only=False, chunk_size=100, max_retries=4, request_timeout=20000)
            log.info(f"Errors are: {errors}")
            return (len(embeddings) - len(errors), None)
        except Exception as e:
            log.warning(f"Failed to insert data: {self.indice} error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return (0, e)
        return (len(embeddings), None)

    def search_embedding(
            self,
            query: list[float],
            k: int = 100,
            filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[tuple[int, float]]: list of k most similar embeddings in (id, score) tuple to the query embedding.
        """
        assert self.client is not None, "should self.init() first"
        log.info("Doing Search............")
        # is_existed_res = self.client.indices.exists(index=self.indice)
        # assert is_existed_res.raw == True, "should self.init() first"
        if filters:
            knn = {
                "size": k,
                "query": {
                    "knn": {
                        self.vector_col_name: {
                            "vector": query,
                            "k": k,
                            "filter": {
                                "range": {self.id_col_name: {"gt": filters["id"]}}
                            }
                        }
                    }
                }
            }
        else:
            knn = {
                "size": k,
                "query": {
                    "knn": {
                        self.vector_col_name: {
                            "vector": query,
                            "k": k
                        }
                    }
                }
            }
        try:
            res = self.client.search(
                index=self.indice,
                body=knn,
                size=k,
                _source=False,
                docvalue_fields=[self.id_col_name],
                stored_fields="_none_",
                filter_path=[f"hits.hits.fields.{self.id_col_name}"],
            )
            res = [h["fields"][self.id_col_name][0] for h in res["hits"]["hits"]]

            return res
        except Exception as e:
            log.warning(f"Failed to search: {self.indice} error: {str(e)}")
            raise e from None

    def optimize(self):
        """optimize will be called between insertion and search in performance cases."""
        assert self.client is not None, "should self.init() first"
        self.client.indices.refresh(index=self.indice)
        force_merge_task_id = self.client.transport.perform_request(method="POST", url=f"/{self.indice}/_forcemerge",
                                                                    params={"wait_for_completion": "false",
                                                                            "max_num_segments": 1})['task']
        log.info(f"Opensearch force merge task id: {force_merge_task_id}")
        SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC = 30
        while True:
            time.sleep(SECONDS_WAITING_FOR_FORCE_MERGE_API_CALL_SEC)
            task_status = self.client.tasks.get(task_id=force_merge_task_id)
            if task_status['completed']:
                break
        # This is required in case the refresh interval was set to -1
        self.client.indices.refresh(index=self.indice)
        # Run warmup
        log.info(f"Running warmup...")
        warmup_response = self.client.transport.perform_request('GET', f'/_plugins/_knn/warmup/{self.indice}', timeout=60)
        log.info(f"Opensearch warmup_response is : {warmup_response}")

    def ready_to_load(self):
        """ready_to_load will be called before load in load cases."""
        pass
