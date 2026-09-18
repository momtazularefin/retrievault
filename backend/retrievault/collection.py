import uuid

COLLECTION_NAME = "retrievault_fastapi"

# The index manifest is stored as a payload-only point in the collection it describes, so the
# service reports the provenance of the index it actually queries. A point without vectors is
# never returned by dense, sparse, or fused search.
MANIFEST_POINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "retrievault:index-manifest"))
