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
    | pruningStmt
    | schedulingStmt
    | resourceConstraintsStmt
    | deploymentConfigStmt
    | pipelineConfigStmt
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

INT8            : 'int8';
FLOAT16         : 'float16';
NONE            : 'none';
BOOL            : 'true' | 'false';

IDENTIFIER      : [a-zA-Z_] [a-zA-Z_0-9]* ;
STRING          : '"' (~["\r\n])* '"' ;
INTEGER         : [0-9]+ ;
NUMBER          : [0-9]+ ('.' [0-9]+)? ;

COMMENT         : '#' .*? '\n' -> skip;
WS              : [ \t\r\n]+ -> skip ;