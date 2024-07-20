-- Source: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#sqlite-schema-reference
CREATE TABLE StringIds (
     -- Consolidation of repetitive string values.

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
     value                       TEXT      NOT NULL                     -- String value.
 );
CREATE TABLE ANALYSIS_FILE (
    -- Analysis file content

    id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID reference value.
    filename                    TEXT      NOT NULL,                    -- File path
    content                     TEXT      NOT NULL,                    -- File content
    globalPid                   INTEGER   NOT NULL                     -- Serialized GlobalId.
 );
 CREATE TABLE ThreadNames (
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Thread name
     priority                    INTEGER,                               -- Priority of the thread.
     globalTid                   INTEGER                                -- Serialized GlobalId.
 );
 CREATE TABLE ProcessStreams (
     globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     filenameId                  INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- File name
     contentId                   INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Stream content
 );
 CREATE TABLE TARGET_INFO_SYSTEM_ENV (
     globalVid                   INTEGER,                               -- Serialized GlobalId.
     devStateName                TEXT      NOT NULL,                    -- Device state name.
     name                        TEXT      NOT NULL,                    -- Property name.
     nameEnum                    INTEGER   NOT NULL,                    -- Property enum value.
     value                       TEXT      NOT NULL                     -- Property value.
 );
 CREATE TABLE TARGET_INFO_NIC_INFO (
     globalId                    INTEGER,                               -- Device state globalId
     stateName                   TEXT      NOT NULL,                    -- Device state name
     nicId                       INTEGER   NOT NULL,                    -- Network interface Id.
     name                        TEXT      NOT NULL,                    -- Network interface name
     deviceId                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_NET_DEVICE_ID(id)
     vendorId                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_NET_VENDOR_ID(id)
     linkLayer                   INTEGER   NOT NULL                     -- REFERENCES ENUM_NET_LINK_TYPE(id)
 );
 CREATE TABLE NIC_ID_MAP (
    -- Map between NIC info nicId and NIC metric globalId

    nicId                       INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NIC_INFO(nicId)
    globalId                    INTEGER   NOT NULL                     -- REFERENCES NET_NIC_METRIC(globalId)
 );
 CREATE TABLE TARGET_INFO_SESSION_START_TIME (
     utcEpochNs                  INTEGER,                               -- UTC Epoch timestamp at start of the capture (ns).
     utcTime                     TEXT,                                  -- Start of the capture in UTC.
     localTime                   TEXT                                   -- Start of the capture in local time of target.
 );
 CREATE TABLE ANALYSIS_DETAILS (
     -- Details about the analysis session.

     globalVid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     duration                    INTEGER   NOT NULL,                    -- The total time span of the entire trace (ns).
     startTime                   INTEGER   NOT NULL,                    -- Trace start timestamp in nanoseconds.
     stopTime                    INTEGER   NOT NULL                     -- Trace stop timestamp in nanoseconds.
 );
 CREATE TABLE PMU_EVENT_REQUESTS (
     -- PMU event requests

     id                          INTEGER   NOT NULL,                    -- PMU event request.
     eventid                     INTEGER,                               -- PMU counter event id.
     source                      INTEGER   NOT NULL,                    -- REFERENCES ENUM_PMU_EVENT_SOURCE(id)
     unit_type                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_PMU_UNIT_TYPE(id)
     event_name                  TEXT,                                  -- PMU counter unique name

     PRIMARY KEY (id)
 );
 CREATE TABLE TARGET_INFO_GPU (
     vmId                        INTEGER   NOT NULL,                    -- Serialized GlobalId.
     id                          INTEGER   NOT NULL,                    -- Device ID.
     name                        TEXT,                                  -- Device name.
     busLocation                 TEXT,                                  -- PCI bus location.
     isDiscrete                  INTEGER,                               -- True if discrete, false if integrated.
     l2CacheSize                 INTEGER,                               -- Size of L2 cache (B).
     totalMemory                 INTEGER,                               -- Total amount of memory on the device (B).
     memoryBandwidth             INTEGER,                               -- Amount of memory transferred (B).
     clockRate                   INTEGER,                               -- Clock frequency (Hz).
     smCount                     INTEGER,                               -- Number of multiprocessors on the device.
     pwGpuId                     INTEGER,                               -- PerfWorks GPU ID.
     uuid                        TEXT,                                  -- Device UUID.
     luid                        INTEGER,                               -- Device LUID.
     chipName                    TEXT,                                  -- Chip name.
     cuDevice                    INTEGER,                               -- CUDA device ID.
     ctxswDevPath                TEXT,                                  -- GPU context switch device node path.
     ctrlDevPath                 TEXT,                                  -- GPU control device node path.
     revision                    INTEGER,                               -- Revision number.
     nodeMask                    INTEGER,                               -- Device node mask.
     constantMemory              INTEGER,                               -- Memory available on device for __constant__ variables (B).
     maxIPC                      INTEGER,                               -- Maximum instructions per count.
     maxRegistersPerBlock        INTEGER,                               -- Maximum number of 32-bit registers available per block.
     maxShmemPerBlock            INTEGER,                               -- Maximum optin shared memory per block.
     maxShmemPerBlockOptin       INTEGER,                               -- Maximum optin shared memory per block.
     maxShmemPerSm               INTEGER,                               -- Maximum shared memory available per multiprocessor (B).
     maxRegistersPerSm           INTEGER,                               -- Maximum number of 32-bit registers available per multiprocessor.
     threadsPerWarp              INTEGER,                               -- Warp size in threads.
     asyncEngines                INTEGER,                               -- Number of asynchronous engines.
     maxWarpsPerSm               INTEGER,                               -- Maximum number of warps per multiprocessor.
     maxBlocksPerSm              INTEGER,                               -- Maximum number of blocks per multiprocessor.
     maxThreadsPerBlock          INTEGER,                               -- Maximum number of threads per block.
     maxBlockDimX                INTEGER,                               -- Maximum X-dimension of a block.
     maxBlockDimY                INTEGER,                               -- Maximum Y-dimension of a block.
     maxBlockDimZ                INTEGER,                               -- Maximum Z-dimension of a block.
     maxGridDimX                 INTEGER,                               -- Maximum X-dimension of a grid.
     maxGridDimY                 INTEGER,                               -- Maximum Y-dimension of a grid.
     maxGridDimZ                 INTEGER,                               -- Maximum Z-dimension of a grid.
     computeMajor                INTEGER,                               -- Major compute capability version number.
     computeMinor                INTEGER,                               -- Minor compute capability version number.
     smMajor                     INTEGER,                               -- Major multiprocessor version number.
     smMinor                     INTEGER                                -- Minor multiprocessor version number.
 );
 CREATE TABLE TARGET_INFO_XMC_SPEC (
     vmId                        INTEGER   NOT NULL,                    -- Serialized GlobalId.
     clientId                    INTEGER   NOT NULL,                    -- Client ID.
     type                        TEXT      NOT NULL,                    -- Client type.
     name                        TEXT      NOT NULL,                    -- Client name.
     groupId                     TEXT      NOT NULL                     -- Client group ID.
 );
 CREATE TABLE TARGET_INFO_PROCESS (
     processId                   INTEGER   NOT NULL,                    -- Process ID.
     openGlVersion               TEXT      NOT NULL,                    -- OpenGL version.
     correlationId               INTEGER   NOT NULL,                    -- Correlation ID of the kernel.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE TARGET_INFO_NVTX_CUDA_DEVICE (
     name                        TEXT      NOT NULL,                    -- CUDA device name assigned using NVTX.
     hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     deviceId                    INTEGER   NOT NULL                     -- Device ID.
 );
 CREATE TABLE TARGET_INFO_NVTX_CUDA_CONTEXT (
     name                        TEXT      NOT NULL,                    -- CUDA context name assigned using NVTX.
     hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     processId                   INTEGER   NOT NULL,                    -- Process ID.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL                     -- Context ID.
 );
 CREATE TABLE TARGET_INFO_NVTX_CUDA_STREAM (
     name                        TEXT      NOT NULL,                    -- CUDA stream name assigned using NVTX.
     hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     processId                   INTEGER   NOT NULL,                    -- Process ID.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL                     -- Stream ID.
 );
 CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (
     nullStreamId                INTEGER   NOT NULL,                    -- Stream ID.
     hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     processId                   INTEGER   NOT NULL,                    -- Process ID.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     parentContextId             INTEGER,                               -- For green context, this is the parent context id.
     isGreenContext              INTEGER                                -- Is this a Green Context?
 );
 CREATE TABLE TARGET_INFO_CUDA_STREAM (
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     hwId                        INTEGER   NOT NULL,                    -- Hardware ID.
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     processId                   INTEGER   NOT NULL,                    -- Process ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     priority                    INTEGER   NOT NULL,                    -- Priority of the stream.
     flag                        INTEGER   NOT NULL                     -- REFERENCES ENUM_CUPTI_STREAM_TYPE(id)
 );
 CREATE TABLE TARGET_INFO_WDDM_CONTEXTS (
     context                     INTEGER   NOT NULL,
     engineType                  INTEGER   NOT NULL,
     nodeOrdinal                 INTEGER   NOT NULL,
     friendlyName                TEXT      NOT NULL
 );
 CREATE TABLE TARGET_INFO_PERF_METRIC (
     id                          INTEGER   NOT NULL,                    -- Event or Metric ID value
     name                        TEXT      NOT NULL,                    -- Event or Metric name
     description                 TEXT      NOT NULL,                    -- Event or Metric description
     unit                        TEXT      NOT NULL                     -- Event or Metric measurement unit
 );
 CREATE TABLE TARGET_INFO_COMPONENT (
     componentId                 INTEGER   NOT NULL,                    -- Component ID value
     name                        TEXT      NOT NULL,                    -- Component name
     instance                    INTEGER   NOT NULL,                    -- Component instance
     parentComponentId           INTEGER   NOT NULL                     -- Parent Component ID value
 );
 CREATE TABLE TARGET_INFO_NETWORK_METRICS (
     metricsListId               INTEGER   NOT NULL,                    -- Metric list ID
     metricsIdx                  INTEGER   NOT NULL,                    -- List index of metric
     name                        TEXT      NOT NULL,                    -- Name of metric
     description                 TEXT      NOT NULL,                    -- Description of metric
     unit                        TEXT      NOT NULL                     -- Measurement unit of metric
 );
 CREATE TABLE META_DATA_CAPTURE (
     -- information about nsys capture parameters

     name                        TEXT      NOT NULL,                    -- Name of meta-data record
     value                       TEXT                                   -- Value of meta-data record
 );
 CREATE TABLE META_DATA_EXPORT (
     -- information about nsys export process

     name                        TEXT      NOT NULL,                    -- Name of meta-data record
     value                       TEXT                                   -- Value of meta-data record
 );
 CREATE TABLE ENUM_NSYS_EVENT_TYPE (
     -- Nsys event type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_NSYS_EVENT_CLASS (
     -- Nsys event class labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_GPU_CTX_SWITCH (
     -- GPU context switch labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_MEMCPY_OPER (
     -- CUDA memcpy operation labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_MEM_KIND (
     -- CUDA memory kind labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_MEMPOOL_TYPE (
     -- CUDA mempool type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_MEMPOOL_OPER (
     -- CUDA mempool operation labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_DEV_MEM_EVENT_OPER (
     -- CUDA device mem event operation labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_KERNEL_LAUNCH_TYPE (
     -- CUDA kernel launch type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG (
     -- CUDA shared memory limit config labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_UNIF_MEM_MIGRATION (
     -- CUDA unified memory migration cause labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_UNIF_MEM_ACCESS_TYPE (
     -- CUDA unified memory access type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUDA_FUNC_CACHE_CONFIG (
     -- CUDA function cache config labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUPTI_STREAM_TYPE (
     -- CUPTI stream type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_CUPTI_SYNC_TYPE (
     -- CUPTI synchronization type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_STACK_UNWIND_METHOD (
     -- Stack unwind method labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_SAMPLING_THREAD_STATE (
     -- Sampling thread state labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENGL_DEBUG_SOURCE (
     -- OpenGL debug source labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENGL_DEBUG_TYPE (
     -- OpenGL debug type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENGL_DEBUG_SEVERITY (
     -- OpenGL debug severity labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_VULKAN_PIPELINE_CREATION_FLAGS (
     -- Vulkan pipeline creation feedback flag labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_D3D12_HEAP_TYPE (
     -- D3D12 heap type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_D3D12_PAGE_PROPERTY (
     -- D3D12 CPU page property labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_D3D12_HEAP_FLAGS (
     -- D3D12 heap flag labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_D3D12_CMD_LIST_TYPE (
     -- D3D12 command list type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENACC_DEVICE (
     -- OpenACC device type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENACC_EVENT_KIND (
     -- OpenACC event type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_EVENT_KIND (
     -- OpenMP event kind labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_THREAD (
     -- OpenMP thread labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_DISPATCH (
     -- OpenMP dispatch labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_SYNC_REGION (
     -- OpenMP sync region labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_WORK (
     -- OpenMP work labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_MUTEX (
     -- OpenMP mutex labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_TASK_FLAG (
     -- OpenMP task flags labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_OPENMP_TASK_STATUS (
     -- OpenMP task status labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_SLI_TRANSER (
     -- OpenMP task status labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_DXGI_FORMAT (
     -- DXGI image format labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_NVDRIVER_EVENT_ID (
     -- NV-Driver event it labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_WDDM_PAGING_QUEUE_TYPE (
     -- WDDM paging queue type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_WDDM_PACKET_TYPE (
     -- WDDM packet type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_WDDM_ENGINE_TYPE (
     -- WDDM engine type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_WDDM_INTERRUPT_TYPE (
     -- WDDM DMA interrupt type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_WDDM_VIDMM_OP_TYPE (
     -- WDDM VidMm operation type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_NET_LINK_TYPE (
     -- NIC link layer labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_NET_DEVICE_ID (
     -- NIC PCIe device id labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_NET_VENDOR_ID (
     -- NIC PCIe vendor id labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_ETW_MEMORY_TRANSFER_TYPE (
     -- memory transfer type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value
     name                        TEXT,                                  -- Enum symbol name
     label                       TEXT                                   -- Enum human name
 );
 CREATE TABLE ENUM_PMU_EVENT_SOURCE (
     -- PMU event source labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_PMU_UNIT_TYPE (
     -- PMU unit type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_VIDEO_ENGINE_TYPE (
     -- Video engine type id labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_VIDEO_ENGINE_CODEC (
     -- Video engine codec labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL (
     -- Diagnostic message severity level labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_DIAGNOSTIC_SOURCE_TYPE (
     -- Diagnostic message source type labels

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE (
     -- Diagnostic message timestamp source lables

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- Enum numerical value.
     name                        TEXT,                                  -- Enum symbol name.
     label                       TEXT                                   -- Enum human name.
 );
 CREATE TABLE GENERIC_EVENT_SOURCES (
     sourceId                    INTEGER   NOT NULL   PRIMARY KEY,      -- Serialized GlobalId.
     data                        TEXT      NOT NULL                     -- JSON encoded generic event source description.
 );
 CREATE TABLE GENERIC_EVENT_TYPES (
     typeId                      INTEGER   NOT NULL   PRIMARY KEY,      -- Serialized GlobalId.
     sourceId                    INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_SOURCES(sourceId)
     data                        TEXT      NOT NULL                     -- JSON encoded generic event type description.
 );
 CREATE TABLE GENERIC_EVENTS (
     -- Events for which the schema is known only at the run-time.

     rawTimestamp                INTEGER   NOT NULL,                    -- Raw event timestamp recorded during profiling.
     timestamp                   INTEGER,                               -- Event timestamp converted to the profiling session timeline.
     typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
     data                        TEXT      NOT NULL                     -- JSON encoded event data.
 );
 CREATE TABLE ETW_PROVIDERS (
     -- Names and identifiers of ETW providers captured in the report.

     providerId                  INTEGER   NOT NULL   PRIMARY KEY,      -- Provider ID.
     providerNameId              INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Provider name
     guid                        TEXT      NOT NULL                     -- ETW Provider GUID.
 );
 CREATE TABLE ETW_TASKS (
     -- Names and identifiers of ETW tasks captured in the report.

     taskNameId                  INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Task name
     taskId                      INTEGER   NOT NULL,                    -- The event task ID.
     providerId                  INTEGER   NOT NULL                     -- Provider ID.
 );
 CREATE TABLE ETW_EVENTS (
     -- Raw ETW events captured in the report.

     timestamp                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     processId                   INTEGER,                               -- Process ID.
     threadId                    INTEGER,                               -- Thread ID.
     providerId                  INTEGER   NOT NULL,                    -- Provider ID.
     taskId                      INTEGER   NOT NULL,                    -- The event task ID.
     eventId                     INTEGER   NOT NULL,                    -- Event ID.
     version                     INTEGER   NOT NULL,                    -- The event version.
     opcode                      INTEGER,                               -- The event opcode.
     data                        TEXT      NOT NULL                     -- JSON encoded event data.
 );
 CREATE TABLE TARGET_INFO_GPU_METRICS (
     -- GPU Metrics, metric names and ids.

     typeId                      INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_TYPES(typeId)
     sourceId                    INTEGER   NOT NULL,                    -- REFERENCES GENERIC_EVENT_SOURCES(sourceId)
     typeName                    TEXT      NOT NULL,                    -- Name of event type.
     metricId                    INTEGER   NOT NULL,                    -- Id of metric in event; not assumed to be stable.
     metricName                  TEXT      NOT NULL                     -- Definitive name of metric.
 );
 CREATE TABLE GPU_METRICS (
     -- GPU Metrics, events and values.

     timestamp                   INTEGER,                               -- Event timestamp (ns).
     typeId                      INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_GPU_METRICS(typeId) and GENERIC_EVENT_TYPES(typeId)
     metricId                    INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_GPU_METRICS(metricId)
     value                       INTEGER   NOT NULL                     -- Counter data value
 );
 CREATE TABLE MPI_COMMUNICATORS (
     -- Identification of MPI communication groups.

     rank                        INTEGER,                               -- Active MPI rank
     timestamp                   INTEGER,                               -- Time of MPI communicator creation.
     commHandle                  INTEGER,                               -- MPI communicator handle
     parentHandle                INTEGER,                               -- MPI communicator handle
     localRank                   INTEGER,                               -- Local MPI rank in a communicator.
     size                        INTEGER,                               -- MPI communicator size.
     groupRoot                   INTEGER,                               -- Root rank (global) in MPI communicator.
     groupRootUid                INTEGER,                               -- Group root's communicator ID.
     members                     TEXT                                   -- MPI communicator members (index is global rank).
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     bytes                       INTEGER   NOT NULL,                    -- Number of bytes transferred (B).
     copyKind                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMCPY_OPER(id)
     deprecatedSrcId             INTEGER,                               -- Deprecated, use srcDeviceId instead.
     srcKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
     dstKind                     INTEGER,                               -- REFERENCES ENUM_CUDA_MEM_KIND(id)
     srcDeviceId                 INTEGER,                               -- Source device ID.
     srcContextId                INTEGER,                               -- Source context ID.
     dstDeviceId                 INTEGER,                               -- Destination device ID.
     dstContextId                INTEGER,                               -- Destination context ID.
     migrationCause              INTEGER,                               -- REFERENCES ENUM_CUDA_UNIF_MEM_MIGRATION(id)
     graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
     virtualAddress              INTEGER                                -- Virtual base address of the page/s being transferred.
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     value                       INTEGER   NOT NULL,                    -- Value assigned to memory.
     bytes                       INTEGER   NOT NULL,                    -- Number of bytes set (B).
     graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
     memKind                     INTEGER                                -- REFERENCES ENUM_CUDA_MEM_KIND(id)
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
     shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
     mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
     launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KERNEL_LAUNCH_TYPE(id)
     cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
     registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
     gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
     gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
     gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
     blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
     blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
     blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
     staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
     dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
     localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
     localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
     gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
     sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
     graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
     sharedMemoryLimitConfig     INTEGER                                -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- Correlation ID of the synchronization API to which this result is associated.
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     syncType                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUPTI_SYNC_TYPE(id)
     eventId                     INTEGER   NOT NULL                     -- Event ID for which the synchronization API is called.
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT (
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- Correlation ID of the event record API to which this result is associated.
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     eventId                     INTEGER   NOT NULL                     -- Event ID for which the event record API is called.
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     greenContextId              INTEGER,                               -- Green context ID.
     streamId                    INTEGER   NOT NULL,                    -- Stream ID.
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     graphId                     INTEGER   NOT NULL,                    -- REFERENCES CUDA_GRAPH_EVENTS(graphId)
     graphExecId                 INTEGER   NOT NULL                     -- REFERENCES CUDA_GRAPH_EVENTS(graphExecId)
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- ID used to identify events that this function call has triggered.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
     callchainId                 INTEGER                                -- REFERENCES CUDA_CALLCHAINS(id)
 );
 CREATE TABLE CUDNN_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE CUBLAS_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE CUDA_GRAPH_NODE_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     graphNodeId                 INTEGER   NOT NULL,                    -- REFERENCES CUDA_GRAPH_NODE_EVENTS(graphNodeId)
     originalGraphNodeId         INTEGER                                -- Reference to the original graph node ID, if cloned node.
 );
 CREATE TABLE CUDA_GRAPH_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     graphId                     INTEGER,                               -- Graph ID.
     originalGraphId             INTEGER,                               -- Reference to the original graph ID, if cloned.
     graphExecId                 INTEGER                                -- Executable graph ID.
 );
 CREATE TABLE CUDA_UM_CPU_PAGE_FAULT_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     address                     INTEGER   NOT NULL,                    -- Virtual address of the page that faulted.
     originalFaultPc             INTEGER,                               -- Program counter of the CPU instruction that caused the page fault.
     CpuInstruction              INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
     unresolvedFaultPc           INTEGER                                -- True if the program counter was not resolved.
 );
 CREATE TABLE CUDA_UM_GPU_PAGE_FAULT_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     address                     INTEGER   NOT NULL,                    -- Virtual address of the page that faulted.
     numberOfPageFaults          INTEGER   NOT NULL,                    -- Number of page faults for the same page.
     faultAccessType             INTEGER   NOT NULL                     -- REFERENCES ENUM_CUDA_UNIF_MEM_ACCESS_TYPE(id)
 );
 CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     address                     INTEGER   NOT NULL,                    -- Virtual address of the allocation/deallocation.
     pc                          INTEGER   NOT NULL,                    -- Program counter of the allocation/deallocation.
     bytes                       INTEGER   NOT NULL,                    -- Number of bytes allocated/deallocated (B).
     memKind                     INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEM_KIND(id)
     memoryOperationType         INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_DEV_MEM_EVENT_OPER(id)
     name                        TEXT,                                  -- Variable name, if available.
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     localMemoryPoolAddress      INTEGER,                               -- Base address of the local memory pool used
     localMemoryPoolReleaseThreshold   INTEGER,                         -- Release threshold of the local memory pool used
     localMemoryPoolSize         INTEGER,                               -- Size of the local memory pool used
     localMemoryPoolUtilizedSize   INTEGER,                             -- Utilized size of the local memory pool used
     importedMemoryPoolAddress   INTEGER,                               -- Base address of the imported memory pool used
     importedMemoryPoolProcessId   INTEGER                              -- Process ID of the imported memory pool used
 );
 CREATE TABLE CUDA_GPU_MEMORY_POOL_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     globalPid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     deviceId                    INTEGER   NOT NULL,                    -- Device ID.
     address                     INTEGER   NOT NULL,                    -- The base virtual address of the memory pool.
     operationType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMPOOL_OPER(id)
     poolType                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_CUDA_MEMPOOL_TYPE(id)
     correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
     minBytesToKeep              INTEGER,                               -- Minimum number of bytes to keep of the memory pool.
     localMemoryPoolReleaseThreshold   INTEGER,                         -- Release threshold of the local memory pool used
     localMemoryPoolSize         INTEGER,                               -- Size of the local memory pool used
     localMemoryPoolUtilizedSize   INTEGER                              -- Utilized size of the local memory pool used
 );
 CREATE TABLE CUDA_CALLCHAINS (
     id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, stackDepth).
     symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
     unresolved                  INTEGER,                               -- True if the symbol was not resolved.
     originalIP                  INTEGER,                               -- Instruction pointer value.
     stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

     PRIMARY KEY (id, stackDepth)
 );
 CREATE TABLE MPI_RANKS (
     -- Mapping of global thread IDs (gtid) to MPI ranks

     globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     rank                        INTEGER   NOT NULL                     -- MPI rank
 );
 CREATE TABLE MPI_P2P_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     commHandle                  INTEGER,                               -- MPI communicator handle
     tag                         INTEGER,                               -- MPI message tag
     remoteRank                  INTEGER,                               -- MPI remote rank (destination or source)
     size                        INTEGER,                               -- MPI message size in bytes
     requestHandle               INTEGER                                -- MPI request handle
 );
 CREATE TABLE MPI_COLLECTIVES_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     commHandle                  INTEGER,                               -- MPI communicator handle
     rootRank                    INTEGER,                               -- root rank in the collective
     size                        INTEGER,                               -- MPI message size in bytes (send size for bidirectional ops)
     recvSize                    INTEGER,                               -- MPI receive size in bytes
     requestHandle               INTEGER                                -- MPI request handle
 );
 CREATE TABLE MPI_START_WAIT_EVENTS (
     -- MPI_Wait* and MPI_Start*

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     requestHandle               INTEGER                                -- MPI request handle
 );
 CREATE TABLE MPI_OTHER_EVENTS (
     -- MPI events without additional parameters

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER                                -- REFERENCES StringIds(id) -- Registered NVTX domain
 );
 CREATE TABLE UCP_WORKERS (
     globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     workerUid                   INTEGER   NOT NULL                     -- UCP worker UID
 );
 CREATE TABLE UCP_SUBMIT_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     bufferAddr                  INTEGER,                               -- Address of the message buffer
     packedSize                  INTEGER,                               -- Message size (packed) in bytes
     peerWorkerUid               INTEGER,                               -- Peer's UCP worker UID
     tag                         INTEGER                                -- UCP message tag
 );
 CREATE TABLE UCP_PROGRESS_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     bufferAddr                  INTEGER,                               -- Address of the message buffer
     packedSize                  INTEGER,                               -- Message size (packed) in bytes
     peerWorkerUid               INTEGER,                               -- Peer's UCP worker UID
     tag                         INTEGER                                -- UCP message tag
 );
 CREATE TABLE UCP_EVENTS (
     -- UCP events without additional parameters

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER                                -- REFERENCES StringIds(id) -- Registered NVTX domain
 );
 CREATE TABLE NVTX_EVENTS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     eventType                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
     rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
     category                    INTEGER,                               -- User-controlled ID that can be used to group events.
     color                       INTEGER,                               -- Encoded ARGB color value.
     text                        TEXT,                                  -- Optional text message for non registered strings.
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     endGlobalTid                INTEGER,                               -- Serialized GlobalId.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain
     domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
     uint64Value                 INTEGER,                               -- One of possible payload value union members.
     int64Value                  INTEGER,                               -- One of possible payload value union members.
     doubleValue                 REAL,                                  -- One of possible payload value union members.
     uint32Value                 INTEGER,                               -- One of possible payload value union members.
     int32Value                  INTEGER,                               -- One of possible payload value union members.
     floatValue                  REAL,                                  -- One of possible payload value union members.
     jsonTextId                  INTEGER,                               -- One of possible payload value union members.
     jsonText                    TEXT                                   -- One of possible payload value union members.
 );
 CREATE TABLE OPENGL_API (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     endGlobalTid                INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- First function name
     endNameId                   INTEGER,                               -- REFERENCES StringIds(id) -- Last function name
     returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     contextId                   INTEGER,                               -- Context ID.
     gpu                         INTEGER,                               -- GPU index.
     display                     INTEGER                                -- Display ID.
 );
 CREATE TABLE OPENGL_WORKLOAD (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     endGlobalTid                INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- First function name
     endNameId                   INTEGER,                               -- REFERENCES StringIds(id) -- Last function name
     returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     contextId                   INTEGER,                               -- Context ID.
     gpu                         INTEGER,                               -- GPU index.
     display                     INTEGER                                -- Display ID.
 );
 CREATE TABLE KHR_DEBUG_EVENTS (
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER,                               -- Event end timestamp (ns).
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Debug marker/group text
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     source                      INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_SOURCE(id)
     khrdType                    INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_TYPE(id)
     id                          INTEGER,                               -- KHR event ID.
     severity                    INTEGER,                               -- REFERENCES ENUM_OPENGL_DEBUG_SEVERITY(id)
     correlationId               INTEGER,                               -- ID used to correlate KHR CPU trace to GPU trace.
     context                     INTEGER                                -- Context ID.
 );
 CREATE TABLE OSRT_API (
     -- OS runtime libraries traced to gather information about low-level userspace APIs.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     returnValue                 INTEGER   NOT NULL,                    -- Return value of the function call.
     nestingLevel                INTEGER,                               -- Zero-base index of the nesting level.
     callchainId                 INTEGER   NOT NULL                     -- REFERENCES OSRT_CALLCHAINS(id)
 );
 CREATE TABLE OSRT_CALLCHAINS (
     -- Callchains attached to OSRT events, depending on selected profiling settings.

     id                          INTEGER   NOT NULL,                    -- Part of PRIMARY KEY (id, stackDepth).
     symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
     kernelMode                  INTEGER,                               -- True if kernel mode.
     thumbCode                   INTEGER,                               -- True if thumb code.
     unresolved                  INTEGER,                               -- True if the symbol was not resolved.
     specialEntry                INTEGER,                               -- True if artifical entry added during processing callchain.
     originalIP                  INTEGER,                               -- Instruction pointer value.
     unwindMethod                INTEGER,                               -- REFERENCES ENUM_STACK_UNWIND_METHOD(id)
     stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

     PRIMARY KEY (id, stackDepth)
 );
 CREATE TABLE PROFILER_OVERHEAD (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     returnValue                 INTEGER   NOT NULL                     -- Return value of the function call.
 );
 CREATE TABLE SCHED_EVENTS (
     -- Thread scheduling events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     cpu                         INTEGER   NOT NULL,                    -- ID of CPU this thread was scheduled in or out.
     isSchedIn                   INTEGER   NOT NULL,                    -- 0 if thread was scheduled out, non-zero otherwise.
     globalTid                   INTEGER                                -- Serialized GlobalId.
 );
 CREATE TABLE COMPOSITE_EVENTS (
     -- Thread sampling events.

     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the composite event.
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     cpu                         INTEGER,                               -- ID of CPU this thread was running on.
     threadState                 INTEGER,                               -- REFERENCES ENUM_SAMPLING_THREAD_STATE(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     cpuCycles                   INTEGER   NOT NULL                     -- Value of Performance Monitoring Unit (PMU) counter.
 );
 CREATE TABLE SAMPLING_CALLCHAINS (
     -- Callchain entries obtained from composite events, used to construct function table views.

     id                          INTEGER   NOT NULL,                    -- REFERENCES COMPOSITE_EVENTS(id)
     symbol                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     module                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Module name
     kernelMode                  INTEGER,                               -- True if kernel mode.
     thumbCode                   INTEGER,                               -- True if thumb code.
     unresolved                  INTEGER,                               -- True if the symbol was not resolved.
     specialEntry                INTEGER,                               -- True if artifical entry added during processing callchain.
     originalIP                  INTEGER,                               -- Instruction pointer value.
     unwindMethod                INTEGER,                               -- REFERENCES ENUM_STACK_UNWIND_METHOD(id)
     stackDepth                  INTEGER   NOT NULL,                    -- Zero-base index of the given function in call stack.

     PRIMARY KEY (id, stackDepth)
 );
 CREATE TABLE PERF_EVENT_CPU_RAW_EVENT (
     -- Sampled CPU Performance Counters.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     vmId                        INTEGER,                               -- VM ID.
     componentId                 INTEGER,                               -- REFERENCES TARGET_INFO_COMPONENT(componentId)
     eventId                     INTEGER,                               -- REFERENCES TARGET_INFO_PERF_METRIC(id)
     count                       INTEGER                                -- Counter data value
 );
 CREATE TABLE PERF_EVENT_CPU_METRIC_EVENT (
     -- CPU Metric values from Sampled Performance Counters.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     vmId                        INTEGER,                               -- VM ID.
     componentId                 INTEGER,                               -- REFERENCES TARGET_INFO_COMPONENT(componentId)
     metricId                    INTEGER,                               -- REFERENCES TARGET_INFO_PERF_METRIC(id)
     value                       REAL                                   -- Metric data value
 );
 CREATE TABLE SLI_QUERIES (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     frameId                     INTEGER   NOT NULL,                    -- Index of the graphics frame starting from 1.
     occQueryIssued              INTEGER   NOT NULL,                    -- Occlusion query issued.
     occQueryAsked               INTEGER   NOT NULL,                    -- Occlusion query asked.
     eventQueryIssued            INTEGER   NOT NULL,                    -- Event query issued.
     eventQueryAsked             INTEGER   NOT NULL,                    -- Event query asked.
     numberOfTransferEvents      INTEGER   NOT NULL,                    -- Number of transfer events.
     amountOfTransferredData     INTEGER   NOT NULL                     -- Cumulative size of resource data that was transferred.
 );
 CREATE TABLE SLI_P2P (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     frameId                     INTEGER   NOT NULL,                    -- Index of the graphics frame starting from 1.
     transferSkipped             INTEGER   NOT NULL,                    -- Number of transfers that were skipped.
     srcGpu                      INTEGER   NOT NULL,                    -- Source GPU ID.
     dstGpu                      INTEGER   NOT NULL,                    -- Destination GPU ID.
     numSubResources             INTEGER   NOT NULL,                    -- Number of sub-resources to transfer.
     resourceSize                INTEGER   NOT NULL,                    -- Size of resource.
     subResourceIdx              INTEGER   NOT NULL,                    -- Sub-resource index.
     smplWidth                   INTEGER,                               -- Sub-resource surface width in samples.
     smplHeight                  INTEGER,                               -- Sub-resource surface height in samples.
     smplDepth                   INTEGER,                               -- Sub-resource surface depth in samples.
     bytesPerElement             INTEGER,                               -- Number of bytes per element.
     dxgiFormat                  INTEGER,                               -- REFERENCES ENUM_DXGI_FORMAT(id)
     logSurfaceNames             TEXT,                                  -- Surface name.
     transferInfo                INTEGER,                               -- REFERENCES ENUM_SLI_TRANSER(id)
     isEarlyPushManagedByNvApi   INTEGER,                               -- True if early push managed by NVAPI. False otherwise.
     useAsyncP2pForResolve       INTEGER,                               -- True if async Peer-to-Peer used for resolve. False otherwise.
     transferFuncName            TEXT,                                  -- "A - BE" for asynchronous transfer, "S - BE" for synchronous transfer.
     regimeName                  TEXT,                                  -- Name of the regime scope that includes the resource.
     debugName                   TEXT,                                  -- Debug name assigned to the resource by the application code.
     bindType                    TEXT                                   -- Bind type.
 );
 CREATE TABLE SLI_STATS (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     countComplexFrames          INTEGER   NOT NULL,                    -- Complex frames count.
     countStats                  INTEGER   NOT NULL,                    -- Number of frame statistics collected for the inactive-time histogram.
     totalInactiveTime           INTEGER   NOT NULL,                    -- Total inactive time (µs).
     minPbSize                   INTEGER   NOT NULL,                    -- Min push buffer size.
     maxPbSize                   INTEGER   NOT NULL,                    -- Max push buffer size.
     totalPbSize                 INTEGER   NOT NULL                     -- Total push buffer size.
 );
 CREATE TABLE DX12_API (
     id                          INTEGER   NOT NULL   PRIMARY KEY,
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     shortContextId              INTEGER,                               -- Short form of the COM interface object address.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
     commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
     objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
     longContextId               INTEGER                                -- Long form of the COM interface object address.
 );
 CREATE TABLE DX12_WORKLOAD (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     shortContextId              INTEGER,                               -- Short form of the COM interface object address.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     gpu                         INTEGER,                               -- GPU index.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
     commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
     objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
     longContextId               INTEGER                                -- Long form of the COM interface object address.
 );
 CREATE TABLE DX12_MEMORY_OPERATION (
     gpu                         INTEGER,                               -- GPU index.
     rangeStart                  INTEGER,                               -- Offset denoting the beginning of a memory range (B).
     rangeEnd                    INTEGER,                               -- Offset denoting the end of a memory range (B).
     subresourceId               INTEGER,                               -- Subresource index.
     heapType                    INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_TYPE(id)
     heapFlags                   INTEGER,                               -- REFERENCES ENUM_D3D12_HEAP_FLAGS(id)
     cpuPageProperty             INTEGER,                               -- REFERENCES ENUM_D3D12_PAGE_PROPERTY(id)
     nvApiFlags                  INTEGER,                               -- NV specific flags. See docs for specifics.
     traceEventId                INTEGER   NOT NULL                     -- REFERENCES DX12_API(id)
 );
 CREATE TABLE VULKAN_API (
     id                          INTEGER   NOT NULL   PRIMARY KEY,
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     contextId                   INTEGER                                -- Short form of the interface object address.
 );
 CREATE TABLE VULKAN_WORKLOAD (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     gpu                         INTEGER,                               -- GPU index.
     contextId                   INTEGER,                               -- Short form of the interface object address.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER                                -- REFERENCES StringIds(id) -- Vulkan CPU debug marker string
 );
 CREATE TABLE VULKAN_DEBUG_API (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     contextId                   INTEGER,                               -- Short form of the interface object address.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER                                -- REFERENCES StringIds(id) -- Vulkan CPU debug marker string
 );
 CREATE TABLE VULKAN_PIPELINE_CREATION_EVENTS (
     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the pipeline creation event.
     duration                    INTEGER,                               -- Event duration (ns).
     flags                       INTEGER,                               -- REFERENCES ENUM_VULKAN_PIPELINE_CREATION_FLAGS(id)
     traceEventId                INTEGER   NOT NULL                     -- REFERENCES VULKAN_API(id) -- ID of the attached vulkan API.
 );
 CREATE TABLE VULKAN_PIPELINE_STAGE_EVENTS (
     id                          INTEGER   NOT NULL   PRIMARY KEY,      -- ID of the pipeline stage event.
     duration                    INTEGER,                               -- Event duration (ns).
     flags                       INTEGER,                               -- REFERENCES ENUM_VULKAN_PIPELINE_CREATION_FLAGS(id)
     creationEventId             INTEGER   NOT NULL                     -- REFERENCES VULKAN_PIPELINE_CREATION_EVENTS(id) -- ID of the attached pipeline creation event.
 );
 CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS (
     tag                         INTEGER   NOT NULL,                    -- REFERENCES ENUM_GPU_CTX_SWITCH(id)
     vmId                        INTEGER   NOT NULL,                    -- VM ID.
     seqNo                       INTEGER   NOT NULL,                    -- Sequential event number.
     contextId                   INTEGER   NOT NULL,                    -- Context ID.
     timestamp                   INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     globalPid                   INTEGER,                               -- Serialized GlobalId.
     gpuId                       INTEGER                                -- GPU index.
 );
 CREATE TABLE OPENMP_EVENT_KIND_THREAD (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     threadId                    INTEGER,                               -- Internal thread sequence starting from 1.
     threadType                  INTEGER                                -- REFERENCES ENUM_OPENMP_THREAD(id)
 );
 CREATE TABLE OPENMP_EVENT_KIND_PARALLEL (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- Internal parallel region sequence starting from 1.
     parentTaskId                INTEGER                                -- ID for task that creates this parallel region.
 );
 CREATE TABLE OPENMP_EVENT_KIND_SYNC_REGION_WAIT (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER,                               -- ID of the task that this event belongs to.
     kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_SYNC_REGION(id)
 );
 CREATE TABLE OPENMP_EVENT_KIND_SYNC_REGION (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER,                               -- ID of the task that this event belongs to.
     kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_SYNC_REGION(id)
 );
 CREATE TABLE OPENMP_EVENT_KIND_TASK (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER,                               -- ID of the task that this event belongs to.
     kind                        INTEGER                                -- REFERENCES ENUM_OPENMP_TASK_FLAG(id)
 );
 CREATE TABLE OPENMP_EVENT_KIND_MASTER (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER                                -- ID of the task that this event belongs to.
 );
 CREATE TABLE OPENMP_EVENT_KIND_REDUCTION (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER                                -- ID of the task that this event belongs to.
 );
 CREATE TABLE OPENMP_EVENT_KIND_TASK_CREATE (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parentTaskId                INTEGER,                               -- ID of the parent task that is creating a new task.
     newTaskId                   INTEGER                                -- ID of the new task that is being created.
 );
 CREATE TABLE OPENMP_EVENT_KIND_TASK_SCHEDULE (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     priorTaskId                 INTEGER,                               -- ID of the task that is being switched out.
     priorTaskStatus             INTEGER,                               -- REFERENCES ENUM_OPENMP_TASK_STATUS(id)
     nextTaskId                  INTEGER                                -- ID of the task that is being switched in.
 );
 CREATE TABLE OPENMP_EVENT_KIND_CANCEL (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     taskId                      INTEGER                                -- ID of the task that is being cancelled.
 );
 CREATE TABLE OPENMP_EVENT_KIND_MUTEX_WAIT (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
     waitId                      INTEGER,                               -- ID indicating the object being waited.
     taskId                      INTEGER                                -- ID of the task that this event belongs to.
 );
 CREATE TABLE OPENMP_EVENT_KIND_CRITICAL_SECTION (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
     waitId                      INTEGER                                -- ID indicating the object being held.
 );
 CREATE TABLE OPENMP_EVENT_KIND_MUTEX_RELEASED (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
     waitId                      INTEGER,                               -- ID indicating the object being released.
     taskId                      INTEGER                                -- ID of the task that this event belongs to.
 );
 CREATE TABLE OPENMP_EVENT_KIND_LOCK_INIT (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
     waitId                      INTEGER                                -- ID indicating object being created/destroyed.
 );
 CREATE TABLE OPENMP_EVENT_KIND_LOCK_DESTROY (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_MUTEX(id)
     waitId                      INTEGER                                -- ID indicating object being created/destroyed.
 );
 CREATE TABLE OPENMP_EVENT_KIND_WORKSHARE (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_WORK(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER,                               -- ID of the task that this event belongs to.
     count                       INTEGER                                -- Measure of the quantity of work involved in the region.
 );
 CREATE TABLE OPENMP_EVENT_KIND_DISPATCH (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     kind                        INTEGER,                               -- REFERENCES ENUM_OPENMP_DISPATCH(id)
     parallelId                  INTEGER,                               -- ID of the parallel region that this event belongs to.
     taskId                      INTEGER                                -- ID of the task that this event belongs to.
 );
 CREATE TABLE OPENMP_EVENT_KIND_FLUSH (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- Currently unused.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     eventKind                   INTEGER,                               -- REFERENCES ENUM_OPENMP_EVENT_KIND(id)
     threadId                    INTEGER                                -- ID of the thread that this event belongs to.
 );
 CREATE TABLE D3D11_PIX_DEBUG_API (
     -- D3D11 debug marker events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     shortContextId              INTEGER,                               -- Short form of the COM interface object address.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER                                -- REFERENCES StringIds(id) -- PIX marker text
 );
 CREATE TABLE D3D12_PIX_DEBUG_API (
     -- D3D12 debug marker events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     correlationId               INTEGER,                               -- First ID matching an API call to GPU workloads.
     endCorrelationId            INTEGER,                               -- Last ID matching an API call to GPU workloads.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     shortContextId              INTEGER,                               -- Short form of the COM interface object address.
     frameId                     INTEGER,                               -- Index of the graphics frame starting from 1.
     color                       INTEGER,                               -- Encoded ARGB color value.
     textId                      INTEGER,                               -- REFERENCES StringIds(id) -- PIX marker text
     commandListType             INTEGER,                               -- REFERENCES ENUM_D3D12_CMD_LIST_TYPE(id)
     objectNameId                INTEGER,                               -- REFERENCES StringIds(id) -- D3D12 object name
     longContextId               INTEGER                                -- Long form of the COM interface object address.
 );
 CREATE TABLE WDDM_EVICT_ALLOCATION_EVENTS (
     -- Raw ETW EvictAllocation events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     allocationHandle            INTEGER   NOT NULL                     -- Global allocation handle.
 );
 CREATE TABLE WDDM_PAGING_QUEUE_PACKET_START_EVENTS (
     -- Raw ETW PagingQueuePacketStart events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     dxgDevice                   INTEGER,                               -- Address of an IDXGIDevice.
     dxgAdapter                  INTEGER,                               -- Address of an IDXGIAdapter.
     pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
     pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
     sequenceId                  INTEGER   NOT NULL,                    -- Internal sequence starting from 0.
     alloc                       INTEGER,                               -- Allocation handle.
     vidMmOpType                 INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_VIDMM_OP_TYPE(id)
     pagingQueueType             INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_PAGING_QUEUE_TYPE(id)
 );
 CREATE TABLE WDDM_PAGING_QUEUE_PACKET_STOP_EVENTS (
     -- Raw ETW PagingQueuePacketStop events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
     pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
     sequenceId                  INTEGER   NOT NULL                     -- Internal sequence starting from 0.
 );
 CREATE TABLE WDDM_PAGING_QUEUE_PACKET_INFO_EVENTS (
     -- Raw ETW PagingQueuePacketInfo events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     pagingQueue                 INTEGER   NOT NULL,                    -- Address of the paging queue.
     pagingQueuePacket           INTEGER   NOT NULL,                    -- Address of the paging queue packet.
     sequenceId                  INTEGER   NOT NULL                     -- Internal sequence starting from 0.
 );
 CREATE TABLE WDDM_QUEUE_PACKET_START_EVENTS (
     -- Raw ETW QueuePacketStart events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     dmaBufferSize               INTEGER   NOT NULL,                    -- The dma buffer size.
     dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
     queuePacket                 INTEGER   NOT NULL,                    -- The address of queue packet.
     progressFenceValue          INTEGER   NOT NULL,                    -- The fence value.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     allocationListSize          INTEGER   NOT NULL,                    -- The number of allocations referenced.
     patchLocationListSize       INTEGER   NOT NULL,                    -- The number of patch locations.
     present                     INTEGER   NOT NULL,                    -- True or False if the packet is a present packet.
     engineType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
     syncObject                  INTEGER                                -- The address of fence object.
 );
 CREATE TABLE WDDM_QUEUE_PACKET_STOP_EVENTS (
     -- Raw ETW QueuePacketStop events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     queuePacket                 INTEGER   NOT NULL,                    -- The address of queue packet.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     preempted                   INTEGER   NOT NULL,                    -- True or False if the packet is preempted.
     timeouted                   INTEGER   NOT NULL,                    -- True or False if the packet is timeouted.
     engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
 );
 CREATE TABLE WDDM_QUEUE_PACKET_INFO_EVENTS (
     -- Raw ETW QueuePacketInfo events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     submitSequence              INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
 );
 CREATE TABLE WDDM_DMA_PACKET_START_EVENTS (
     -- Raw ETW DmaPacketStart events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     queuePacketContext          INTEGER   NOT NULL,                    -- The queue packet context.
     uliSubmissionId             INTEGER   NOT NULL,                    -- The queue packet submission ID.
     dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     quantumStatus               INTEGER   NOT NULL,                    -- The quantum Status.
     engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
 );
 CREATE TABLE WDDM_DMA_PACKET_STOP_EVENTS (
     -- Raw ETW DmaPacketStop events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     uliCompletionId             INTEGER   NOT NULL,                    -- The queue packet completion ID.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     preempted                   INTEGER   NOT NULL,                    -- True or False if the packet is preempted.
     engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
 );
 CREATE TABLE WDDM_DMA_PACKET_INFO_EVENTS (
     -- Raw ETW DmaPacketInfo events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     uliCompletionId             INTEGER   NOT NULL,                    -- The queue packet completion ID.
     faultedVirtualAddress       INTEGER   NOT NULL,                    -- The virtual address of faulted process.
     faultedProcessHandle        INTEGER   NOT NULL,                    -- The address of faulted process.
     packetType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_PACKET_TYPE(id)
     ulQueueSubmitSequence       INTEGER   NOT NULL,                    -- Internal sequence starting from 1.
     interruptType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_WDDM_INTERRUPT_TYPE(id)
     quantumStatus               INTEGER   NOT NULL,                    -- The quantum Status.
     pageFaultFlags              INTEGER   NOT NULL,                    -- The page fault flag ID.
     engineType                  INTEGER   NOT NULL                     -- REFERENCES ENUM_WDDM_ENGINE_TYPE(id)
 );
 CREATE TABLE WDDM_HW_QUEUE_EVENTS (
     -- Raw ETW HwQueueStart events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     context                     INTEGER   NOT NULL,                    -- The context ID of WDDM queue.
     hwQueue                     INTEGER   NOT NULL,                    -- The address of HW queue.
     parentDxgHwQueue            INTEGER   NOT NULL                     -- The address of parent Dxg HW queue.
 );
 CREATE TABLE NVVIDEO_ENCODER_API (
     -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK Encoder APIs.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE NVVIDEO_DECODER_API (
     -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK Decoder APIs.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE NVVIDEO_JPEG_API (
     -- NV Video Encoder API traced to gather information about NVIDIA Video Codek SDK JPEG APIs.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL                     -- REFERENCES StringIds(id) -- Function name
 );
 CREATE TABLE MEMORY_TRANSFER_EVENTS (
     -- Raw ETW Memory Transfer events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     gpu                         INTEGER,                               -- GPU index.
     taskId                      INTEGER   NOT NULL,                    -- The event task ID.
     eventId                     INTEGER   NOT NULL,                    -- Event ID.
     allocationGlobalHandle      INTEGER   NOT NULL,                    -- Address of the global allocation handle.
     dmaBuffer                   INTEGER   NOT NULL,                    -- The reported address of dma buffer.
     size                        INTEGER   NOT NULL,                    -- The size of the dma buffer in bytes.
     offset                      INTEGER   NOT NULL,                    -- The offset from the start of the reported dma buffer in bytes.
     memoryTransferType          INTEGER   NOT NULL                     -- REFERENCES ENUM_ETW_MEMORY_TRANSFER_TYPE(id)
 );
 CREATE TABLE NV_LOAD_BALANCE_MASTER_EVENTS (
     -- Raw ETW NV-wgf2um LoadBalanceMaster events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     eventId                     INTEGER   NOT NULL,                    -- Event ID.
     task                        TEXT      NOT NULL,                    -- The task name.
     frameCount                  INTEGER   NOT NULL,                    -- The frame ID.
     frameTime                   REAL      NOT NULL,                    -- Frame duration.
     averageFrameTime            REAL      NOT NULL,                    -- Average of frame duration.
     averageLatency              REAL      NOT NULL,                    -- Average of latency.
     minLatency                  REAL      NOT NULL,                    -- The minimum latency.
     averageQueuedFrames         REAL      NOT NULL,                    -- Average number of queued frames.
     totalActiveMs               REAL      NOT NULL,                    -- Total active time in milliseconds.
     totalIdleMs                 REAL      NOT NULL,                    -- Total idle time in milliseconds.
     idlePercent                 REAL      NOT NULL,                    -- The percentage of idle time.
     isGPUAlmostOneFrameAhead    INTEGER   NOT NULL                     -- True or False if GPU is almost one frame ahead.
 );
 CREATE TABLE NV_LOAD_BALANCE_EVENTS (
     -- Raw ETW NV-wgf2um LoadBalance events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalTid                   INTEGER   NOT NULL,                    -- Serialized GlobalId.
     gpu                         INTEGER   NOT NULL,                    -- GPU index.
     eventId                     INTEGER   NOT NULL,                    -- Event ID.
     task                        TEXT      NOT NULL,                    -- The task name.
     averageFPS                  REAL      NOT NULL,                    -- Average frame per second.
     queuedFrames                REAL      NOT NULL,                    -- The amount of queued frames.
     averageQueuedFrames         REAL      NOT NULL,                    -- Average number of queued frames.
     currentCPUTime              REAL      NOT NULL,                    -- The current CPU time.
     averageCPUTime              REAL      NOT NULL,                    -- Average CPU time.
     averageStallTime            REAL      NOT NULL,                    -- Average of stall time.
     averageCPUIdleTime          REAL      NOT NULL,                    -- Average CPU idle time.
     isGPUAlmostOneFrameAhead    INTEGER   NOT NULL                     -- True or False if GPU is almost one frame ahead.
 );
 CREATE TABLE PROCESSES (
     -- Names and identifiers of processes captured in the report.

     globalPid                   INTEGER,                               -- Serialized GlobalId.
     pid                         INTEGER,                               -- The process ID.
     name                        TEXT                                   -- The process name.
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_DATA (
     -- OpenACC data events collected using CUPTI.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
     DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
     lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
     cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
     cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
     cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
     srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
     funcName                    INTEGER,                               -- REFERENCES StringIds(id) -- Function in which event occurred
     bytes                       INTEGER,                               -- Number of bytes.
     varName                     INTEGER                                -- REFERENCES StringIds(id) -- Variable name
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH (
     -- OpenACC launch events collected using CUPTI.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
     DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
     lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
     cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
     cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
     cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
     srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
     funcName                    INTEGER,                               -- REFERENCES StringIds(id) -- Function in which event occurred
     numGangs                    INTEGER,                               -- Number of gangs created for this kernel launch.
     numWorkers                  INTEGER,                               -- Number of workers created for this kernel launch.
     vectorLength                INTEGER,                               -- Number of vector lanes created for this kernel launch.
     kernelName                  INTEGER                                -- REFERENCES StringIds(id) -- Kernel name
 );
 CREATE TABLE CUPTI_ACTIVITY_KIND_OPENACC_OTHER (
     -- OpenACC other events collected using CUPTI.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Event name
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     eventKind                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_EVENT_KIND(id)
     DeviceType                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_OPENACC_DEVICE(id)
     lineNo                      INTEGER   NOT NULL,                    -- Line number of the directive or program construct.
     cuDeviceId                  INTEGER   NOT NULL,                    -- CUDA device ID. Valid only if deviceType is acc_device_nvidia.
     cuContextId                 INTEGER   NOT NULL,                    -- CUDA context ID. Valid only if deviceType is acc_device_nvidia.
     cuStreamId                  INTEGER   NOT NULL,                    -- CUDA stream ID. Valid only if deviceType is acc_device_nvidia.
     srcFile                     INTEGER,                               -- REFERENCES StringIds(id) -- Source file name or path
     funcName                    INTEGER                                -- REFERENCES StringIds(id) -- Function in which event occurred
 );
 CREATE TABLE NET_NIC_METRIC (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalId                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
     metricsListId               INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsListId)
     metricsIdx                  INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsIdx)
     value                       INTEGER   NOT NULL                     -- Counter data value
 );
 CREATE TABLE NET_IB_SWITCH_METRIC (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalId                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
     metricsListId               INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsListId)
     metricsIdx                  INTEGER   NOT NULL,                    -- REFERENCES TARGET_INFO_NETWORK_METRICS(metricsIdx)
     value                       INTEGER   NOT NULL                     -- Counter data value
 );
 CREATE TABLE PMU_EVENTS (
     -- CPU Core events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalVm                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
     cpu                         INTEGER   NOT NULL,                    -- CPU ID
     counter_id                  INTEGER                                -- REFERENCES PMU_EVENT_COUNTERS(id)
 );
 CREATE TABLE PMU_EVENT_COUNTERS (
     -- CPU Core events counters.

     id                          INTEGER   NOT NULL,
     idx                         INTEGER   NOT NULL,                    -- REFERENCES PMU_EVENT_REQUESTS(id).
     value                       INTEGER   NOT NULL                     -- Counter data value
 );
 CREATE TABLE TRACE_PROCESS_EVENT_NVMEDIA (
     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     eventClass                  INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_CLASS(id)
     globalTid                   INTEGER,                               -- Serialized GlobalId.
     nameId                      INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Function name
     correlationId               INTEGER                                -- First ID matching an API call to GPU workloads.
 );
 CREATE TABLE UNCORE_PMU_EVENTS (
     -- PMU Uncore events.

     start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
     end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
     globalVm                    INTEGER   NOT NULL,                    -- Serialized GlobalId.
     clusterId                   INTEGER,                               -- Cluster ID.
     counterId                   INTEGER                                -- REFERENCES UNCORE_PMU_EVENT_VALUES(id).
 );
 CREATE TABLE UNCORE_PMU_EVENT_VALUES (
     -- Uncore events values.

     id                          INTEGER   NOT NULL,
     type                        INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
     value                       INTEGER   NOT NULL,                    -- Event value.
     rawId                       INTEGER   NOT NULL,                    -- Event value raw ID.
     clusterId                   INTEGER                                -- Cluster ID.
 );
 CREATE TABLE DIAGNOSTIC_EVENT (
     timestamp                   INTEGER   NOT NULL,                    -- Event timestamp (ns).
     timestampType               INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE(id)
     source                      INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_SOURCE_TYPE(id)
     severity                    INTEGER   NOT NULL,                    -- REFERENCES ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id)
     text                        TEXT      NOT NULL,                    -- Diagnostic message text
     globalPid                   INTEGER                                -- Serialized GlobalId.
 );