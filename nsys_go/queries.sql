-- name: GetCUDAMemoryUsageEvents :many
SELECT
    start,
    globalPid,
    deviceId,
    contextId,
    address,
    pc,
    bytes,
    memKind,
    memoryOperationType,
    name,
    correlationId,
    localMemoryPoolAddress,
    localMemoryPoolReleaseThreshold,
    localMemoryPoolSize,
    localMemoryPoolUtilizedSize,
    importedMemoryPoolAddress,
    importedMemoryPoolProcessId
FROM
    CUDA_GPU_MEMORY_USAGE_EVENTS
LIMIT 100; 
