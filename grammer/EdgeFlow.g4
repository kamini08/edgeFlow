grammar EdgeFlow;

// Parser rules

program
    : statement+ EOF
    ;

statement
    : modelStmt
    | quantizeStmt
    | targetDeviceStmt
    | deployPathStmt
    | inputStreamStmt
    | bufferSizeStmt
    | optimizeForStmt
    | memoryLimitStmt
    | fusionStmt
    | frameworkStmt
    | hybridOptimizationStmt
    | pytorchQuantizeStmt
    | fineTuningStmt
    | pruningStmt
    | schedulingStmt
    | resourceConstraintsStmt
    | deploymentConfigStmt
    | pipelineConfigStmt
    | pipelineStmt
    ;

modelStmt
    : MODEL '=' STRING
    ;

quantizeStmt
    : QUANTIZE '=' quantType
    ;

quantType
    : INT8
    | FLOAT16
    | NONE
    ;

targetDeviceStmt
    : TARGET_DEVICE '=' IDENTIFIER
    ;

deployPathStmt
    : DEPLOY_PATH '=' STRING
    ;

inputStreamStmt
    : INPUT_STREAM '=' IDENTIFIER
    ;

bufferSizeStmt
    : BUFFER_SIZE '=' INTEGER
    ;

optimizeForStmt
    : OPTIMIZE_FOR '=' IDENTIFIER
    ;

memoryLimitStmt
    : MEMORY_LIMIT '=' INTEGER
    ;

fusionStmt
    : FUSION '=' BOOL
    ;

frameworkStmt
    : FRAMEWORK '=' IDENTIFIER
    ;

hybridOptimizationStmt
    : HYBRID_OPTIMIZATION '=' BOOL
    ;

pytorchQuantizeStmt
    : PYTORCH_QUANTIZE '=' pytorchQuantType
    ;

pytorchQuantType
    : DYNAMIC_INT8
    | STATIC_INT8
    | NONE
    ;

fineTuningStmt
    : FINE_TUNING '=' BOOL
    ;

pruningStmt
    : PRUNING '=' pruningConfig
    ;

pruningConfig
    : BOOL
    | BOOL ',' PRUNING_SPARSITY '=' NUMBER
    ;

schedulingStmt
    : SCHEDULING '=' schedulingConfig
    ;

schedulingConfig
    : IDENTIFIER
    | IDENTIFIER ',' SCHEDULING_PARAMS '=' '{' schedulingParamList '}'
    ;

schedulingParamList
    : schedulingParam (',' schedulingParam)*
    ;

schedulingParam
    : IDENTIFIER '=' (STRING | NUMBER | BOOL)
    ;

resourceConstraintsStmt
    : RESOURCE_CONSTRAINTS '=' '{' constraintList '}'
    ;

constraintList
    : constraint (',' constraint)*
    ;

constraint
    : MEMORY_LIMIT '=' NUMBER
    | CPU_LIMIT '=' NUMBER
    | POWER_LIMIT '=' NUMBER
    | LATENCY_LIMIT '=' NUMBER
    ;

deploymentConfigStmt
    : DEPLOYMENT_CONFIG '=' '{' deploymentParamList '}'
    ;

deploymentParamList
    : deploymentParam (',' deploymentParam)*
    ;

deploymentParam
    : DEPLOY_PATH '=' STRING
    | ENVIRONMENT '=' IDENTIFIER
    | RUNTIME '=' IDENTIFIER
    | DEPLOYMENT_MODE '=' IDENTIFIER
    ;

pipelineConfigStmt
    : PIPELINE_CONFIG '=' '{' pipelineParamList '}'
    ;

pipelineParamList
    : pipelineParam (',' pipelineParam)*
    ;

pipelineParam
    : BUFFER_SIZE '=' NUMBER
    | STREAMING_MODE '=' BOOL
    | BATCH_SIZE '=' NUMBER
    | PARALLEL_WORKERS '=' NUMBER
    ;

// ----------------------------------------------------------------------------
// Pipeline-centric constructs (non-breaking additions)
// ----------------------------------------------------------------------------

pipelineStmt
    : PIPELINE IDENTIFIER '(' pipelineAttrList? ')' '{' pipelineBody* '}'
    ;

pipelineAttrList
    : pipelineAttr (',' pipelineAttr)*
    ;

pipelineAttr
    : TARGET '=' IDENTIFIER
    | MEMORY_LIMIT '=' NUMBER
    | FLAGS '=' '[' IDENTIFIER (',' IDENTIFIER)* ']'
    ;

pipelineBody
    : declInput
    | declOutput
    | layerDecl
    | connectionStmt
    ;

declInput
    : INPUT IDENTIFIER ':' tensorType
    ;

declOutput
    : OUTPUT IDENTIFIER ':' tensorType
    ;

tensorType
    : TENSOR '<' DTYPE ',' '[' dimList? ']' '>'
    ;

dimList
    : dim (',' dim)*
    ;

dim
    : INTEGER
    | STAR
    ;

layerDecl
    : IDENTIFIER ':' layerType '(' argList? ')' (ARROW IDENTIFIER)?
    ;

// Type-constrained layer definitions
layerType
    : 'Conv2D'
    | 'Conv1D' 
    | 'Dense'
    | 'MaxPool2D'
    | 'AvgPool2D'
    | 'Flatten'
    | 'Dropout'
    | 'BatchNorm'
    | 'LayerNorm'
    | 'Activation'
    | 'LSTM'
    | 'GRU'
    | 'Embedding'
    | 'Attention'
    ;

argList
    : arg (',' arg)*
    ;

arg
    : IDENTIFIER '=' argValue
    ;

// Type-constrained argument values
argValue
    : STRING                    // For string parameters
    | constrainedNumber        // For numeric parameters with constraints
    | BOOL                     // For boolean parameters
    | activationType           // For activation parameters
    | paddingType             // For padding parameters
    | IDENTIFIER              // For identifiers/references
    | '[' valueList? ']'      // For array parameters
    ;

// Constrained numeric values
constrainedNumber
    : positiveInt             // Positive integers (filters, units, etc.)
    | kernelSize              // Kernel size constraints
    | strideValue             // Stride constraints
    | poolSize                // Pool size constraints
    | dropoutRate             // Dropout rate constraints
    | NUMBER                  // General numbers
    ;

// Positive integers (1 to max)
positiveInt
    : POSITIVE_INT
    ;

// Kernel size constraints (1-15 for most layers)
kernelSize
    : KERNEL_SIZE_INT
    | '(' KERNEL_SIZE_INT ',' KERNEL_SIZE_INT ')'
    ;

// Stride constraints (1-8)
strideValue
    : STRIDE_INT
    | '(' STRIDE_INT ',' STRIDE_INT ')'
    ;

// Pool size constraints (1-8)
poolSize
    : POOL_SIZE_INT
    | '(' POOL_SIZE_INT ',' POOL_SIZE_INT ')'
    ;

// Dropout rate (0.0-0.9)
dropoutRate
    : DROPOUT_RATE
    ;

// Activation type enumeration
activationType
    : 'relu' | 'sigmoid' | 'tanh' | 'softmax' 
    | 'leaky_relu' | 'swish' | 'gelu' | 'linear'
    ;

// Padding type enumeration  
paddingType
    : 'valid' | 'same'
    ;

valueList
    : (STRING | NUMBER | BOOL | IDENTIFIER) (',' (STRING | NUMBER | BOOL | IDENTIFIER))*
    ;

connectionStmt
    : CONNECT IDENTIFIER ARROW IDENTIFIER
    ;

// Lexer rules

MODEL           : 'model';
QUANTIZE        : 'quantize';
TARGET_DEVICE   : 'target_device';
DEPLOY_PATH     : 'deploy_path';
INPUT_STREAM    : 'input_stream';
BUFFER_SIZE     : 'buffer_size';
OPTIMIZE_FOR    : 'optimize_for';
MEMORY_LIMIT    : 'memory_limit';
FUSION          : 'enable_fusion';
FRAMEWORK       : 'framework';
HYBRID_OPTIMIZATION : 'enable_hybrid_optimization';
PYTORCH_QUANTIZE : 'pytorch_quantize';
DYNAMIC_INT8    : 'dynamic_int8';
STATIC_INT8     : 'static_int8';
FINE_TUNING     : 'fine_tuning';
PRUNING         : 'enable_pruning';
PRUNING_SPARSITY : 'pruning_sparsity';
SCHEDULING      : 'scheduling';
SCHEDULING_PARAMS : 'scheduling_params';
RESOURCE_CONSTRAINTS : 'resource_constraints';
DEPLOYMENT_CONFIG : 'deployment_config';
PIPELINE_CONFIG : 'pipeline_config';
CPU_LIMIT       : 'cpu_limit';
POWER_LIMIT     : 'power_limit';
LATENCY_LIMIT   : 'latency_limit';
ENVIRONMENT     : 'environment';
RUNTIME         : 'runtime';
DEPLOYMENT_MODE : 'deployment_mode';
STREAMING_MODE  : 'streaming_mode';
BATCH_SIZE      : 'batch_size';
PARALLEL_WORKERS : 'parallel_workers';

// New tokens for pipeline constructs
PIPELINE        : 'pipeline';
TARGET          : 'target';
FLAGS           : 'flags';
INPUT           : 'input';
OUTPUT          : 'output';
TENSOR          : 'Tensor';
DTYPE           : 'int8' | 'uint8' | 'int16' | 'float16' | 'float32' | 'int32' | 'bool';
CONNECT         : 'connect';
ARROW           : '->';
STAR            : '*';

INT8            : 'int8';
FLOAT16         : 'float16';
NONE            : 'none';
BOOL            : 'true' | 'false';

// Type-constrained tokens
POSITIVE_INT    : [1-9] [0-9]* ;                    // Positive integers only
KERNEL_SIZE_INT : [1-9] | '1' [0-5] ;              // 1-15 for kernel sizes
STRIDE_INT      : [1-8] ;                           // 1-8 for strides  
POOL_SIZE_INT   : [1-8] ;                           // 1-8 for pool sizes
DROPOUT_RATE    : '0.' [0-9] | '0.9' ;              // 0.0-0.9 for dropout

IDENTIFIER      : [a-zA-Z_] [a-zA-Z_0-9]* ;
STRING          : '"' (~["\r\n])* '"' ;
INTEGER         : [0-9]+ ;
NUMBER          : [0-9]+ ('.' [0-9]+)? ;

COMMENT         : '#' .*? '\n' -> skip;
WS              : [ \t\r\n]+ -> skip ;