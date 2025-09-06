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

INT8            : 'int8';
FLOAT16         : 'float16';
NONE            : 'none';
BOOL            : 'true' | 'false';

IDENTIFIER      : [a-zA-Z_] [a-zA-Z_0-9]* ;
STRING          : '"' (~["\r\n])* '"' ;
INTEGER         : [0-9]+ ;

COMMENT         : '#' .*? '\n' -> skip;
WS              : [ \t\r\n]+ -> skip ;