from pydantic import SecretStr, BaseModel

from ..api import DBConfig, DBCaseConfig, MetricType


class OpensearchConfig(DBConfig, BaseModel):
    user: SecretStr
    password: SecretStr
    url: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "basic_auth": (self.user.get_secret_value(), self.password.get_secret_value()),
        }


class OpensearchIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    efConstruction: int | None = None
    efSearch: int | None = None
    M: int | None = None
    dimensions: int | None = None
    primaries: int | None = None
    replicas: int | None = None
    engine: str | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2"
        elif self.metric_type == MetricType.IP:
            return "innerproduct"
        return "cosinesimil"

    def index_param(self) -> dict:
        params = {}
        if self.engine == "nmslib":
            params = {
                "type": "knn_vector",
                "dimension": self.dimensions,
                "method": {
                    "space_type": self.parse_metric(),
                    "name": "hnsw",
                    "engine": "nmslib",
                    "parameters": {
                        "m": self.M,
                        "ef_construction": self.efConstruction
                    }
                },
            }
        elif self.engine == "faiss":
            params = {
                "type": "knn_vector",
                "dimension": self.dimensions,
                "method": {
                    "space_type": self.parse_metric(),
                    "name": "hnsw",
                    "engine": "faiss",
                    "parameters": {
                        "m": self.M,
                        "ef_construction": self.efConstruction,
                        "ef_search": self.efSearch
                    }
                },
            }
        elif self.engine == "lucene":
            params = {
                "type": "knn_vector",
                "dimension": self.dimensions,
                "method": {
                    "space_type": self.parse_metric(),
                    "name": "hnsw",
                    "engine": "lucene",
                    "parameters": {
                        "m": self.M,
                        "ef_construction": self.efConstruction
                    }
                },
            }
        return params

    def search_param(self) -> dict:
        return {
        }
