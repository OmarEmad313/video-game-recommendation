��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
-Adam/recommender_net/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*>
shared_name/-Adam/recommender_net/embedding_3/embeddings/v
�
AAdam/recommender_net/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_3/embeddings/v*
_output_shapes
:	�*
dtype0
�
-Adam/recommender_net/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*>
shared_name/-Adam/recommender_net/embedding_2/embeddings/v
�
AAdam/recommender_net/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_2/embeddings/v*
_output_shapes
:	�2*
dtype0
�
-Adam/recommender_net/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)*>
shared_name/-Adam/recommender_net/embedding_1/embeddings/v
�
AAdam/recommender_net/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_1/embeddings/v*
_output_shapes
:	�)*
dtype0
�
+Adam/recommender_net/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)2*<
shared_name-+Adam/recommender_net/embedding/embeddings/v
�
?Adam/recommender_net/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp+Adam/recommender_net/embedding/embeddings/v*
_output_shapes
:	�)2*
dtype0
�
-Adam/recommender_net/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*>
shared_name/-Adam/recommender_net/embedding_3/embeddings/m
�
AAdam/recommender_net/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_3/embeddings/m*
_output_shapes
:	�*
dtype0
�
-Adam/recommender_net/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*>
shared_name/-Adam/recommender_net/embedding_2/embeddings/m
�
AAdam/recommender_net/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_2/embeddings/m*
_output_shapes
:	�2*
dtype0
�
-Adam/recommender_net/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)*>
shared_name/-Adam/recommender_net/embedding_1/embeddings/m
�
AAdam/recommender_net/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp-Adam/recommender_net/embedding_1/embeddings/m*
_output_shapes
:	�)*
dtype0
�
+Adam/recommender_net/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)2*<
shared_name-+Adam/recommender_net/embedding/embeddings/m
�
?Adam/recommender_net/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp+Adam/recommender_net/embedding/embeddings/m*
_output_shapes
:	�)2*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
&recommender_net/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*7
shared_name(&recommender_net/embedding_3/embeddings
�
:recommender_net/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_3/embeddings*
_output_shapes
:	�*
dtype0
�
&recommender_net/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*7
shared_name(&recommender_net/embedding_2/embeddings
�
:recommender_net/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_2/embeddings*
_output_shapes
:	�2*
dtype0
�
&recommender_net/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)*7
shared_name(&recommender_net/embedding_1/embeddings
�
:recommender_net/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp&recommender_net/embedding_1/embeddings*
_output_shapes
:	�)*
dtype0
�
$recommender_net/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�)2*5
shared_name&$recommender_net/embedding/embeddings
�
8recommender_net/embedding/embeddings/Read/ReadVariableOpReadVariableOp$recommender_net/embedding/embeddings*
_output_shapes
:	�)2*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$recommender_net/embedding/embeddings&recommender_net/embedding_1/embeddings&recommender_net/embedding_2/embeddings&recommender_net/embedding_3/embeddings*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_12519

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
		user_bias

game_embedding
	game_bias
	optimizer

signatures*
 
0
1
2
3*
 
0
1
2
3*

0
1* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

embeddings*
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

embeddings*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

embeddings*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

embeddings*
�
5iter

6beta_1

7beta_2
	8decay
9learning_ratem^m_m`mavbvcvdve*

:serving_default* 
d^
VARIABLE_VALUE$recommender_net/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_2/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&recommender_net/embedding_3/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE*

;trace_0* 

<trace_0* 
* 
 
0
	1

2
3*

=0*
* 
* 
* 
* 
* 
* 

0*

0*
	
0* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 

0*

0*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 

0*

0*
	
0* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 

0*

0*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
8
Z	variables
[	keras_api
	\total
	]count*
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1*

Z	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/recommender_net/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_1/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_2/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_3/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE+Adam/recommender_net/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_1/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_2/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE-Adam/recommender_net/embedding_3/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8recommender_net/embedding/embeddings/Read/ReadVariableOp:recommender_net/embedding_1/embeddings/Read/ReadVariableOp:recommender_net/embedding_2/embeddings/Read/ReadVariableOp:recommender_net/embedding_3/embeddings/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp?Adam/recommender_net/embedding/embeddings/m/Read/ReadVariableOpAAdam/recommender_net/embedding_1/embeddings/m/Read/ReadVariableOpAAdam/recommender_net/embedding_2/embeddings/m/Read/ReadVariableOpAAdam/recommender_net/embedding_3/embeddings/m/Read/ReadVariableOp?Adam/recommender_net/embedding/embeddings/v/Read/ReadVariableOpAAdam/recommender_net/embedding_1/embeddings/v/Read/ReadVariableOpAAdam/recommender_net/embedding_2/embeddings/v/Read/ReadVariableOpAAdam/recommender_net/embedding_3/embeddings/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_12797
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$recommender_net/embedding/embeddings&recommender_net/embedding_1/embeddings&recommender_net/embedding_2/embeddings&recommender_net/embedding_3/embeddings	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount+Adam/recommender_net/embedding/embeddings/m-Adam/recommender_net/embedding_1/embeddings/m-Adam/recommender_net/embedding_2/embeddings/m-Adam/recommender_net/embedding_3/embeddings/m+Adam/recommender_net/embedding/embeddings/v-Adam/recommender_net/embedding_1/embeddings/v-Adam/recommender_net/embedding_2/embeddings/v-Adam/recommender_net/embedding_3/embeddings/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_12864�
�[
�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12355

inputs	"
embedding_12249:	�)2$
embedding_1_12266:	�)$
embedding_2_12287:	�2$
embedding_3_12304:	�
identity��!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12249*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12248f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_12266*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_12265f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_12287*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_12286f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_12304*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_12303_
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       Q
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB i
Tensordot/ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	Transpose*embedding/StatefulPartitionedCall:output:0Tensordot/concat:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������a
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB m
Tensordot/Shape_1Shape,embedding_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:[
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1:output:0Tensordot/free_1:output:0"Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1:output:0Tensordot/axes_1:output:0"Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_2ProdTensordot/GatherV2_2:output:0Tensordot/Const_2:output:0*
T0*
_output_shapes
: [
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_3ProdTensordot/GatherV2_3:output:0Tensordot/Const_3:output:0*
T0*
_output_shapes
: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/axes_1:output:0Tensordot/free_1:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
Tensordot/stack_1PackTensordot/Prod_3:output:0Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose_1	Transpose,embedding_2/StatefulPartitionedCall:output:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������Y
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_2ConcatV2Tensordot/GatherV2:output:0Tensordot/GatherV2_2:output:0 Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: n
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
addAddV2Tensordot:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������w
add_1AddV2add:z:0,embedding_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_12249*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_2_12287*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_embedding_3_layer_call_and_return_conditional_losses_12303

inputs	)
embedding_lookup_12297:	�
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_12297inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12297*'
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12297*'
_output_shapes
:���������}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�2
�

__inference__traced_save_12797
file_prefixC
?savev2_recommender_net_embedding_embeddings_read_readvariableopE
Asavev2_recommender_net_embedding_1_embeddings_read_readvariableopE
Asavev2_recommender_net_embedding_2_embeddings_read_readvariableopE
Asavev2_recommender_net_embedding_3_embeddings_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopJ
Fsavev2_adam_recommender_net_embedding_embeddings_m_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_1_embeddings_m_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_2_embeddings_m_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_3_embeddings_m_read_readvariableopJ
Fsavev2_adam_recommender_net_embedding_embeddings_v_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_1_embeddings_v_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_2_embeddings_v_read_readvariableopL
Hsavev2_adam_recommender_net_embedding_3_embeddings_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_recommender_net_embedding_embeddings_read_readvariableopAsavev2_recommender_net_embedding_1_embeddings_read_readvariableopAsavev2_recommender_net_embedding_2_embeddings_read_readvariableopAsavev2_recommender_net_embedding_3_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopFsavev2_adam_recommender_net_embedding_embeddings_m_read_readvariableopHsavev2_adam_recommender_net_embedding_1_embeddings_m_read_readvariableopHsavev2_adam_recommender_net_embedding_2_embeddings_m_read_readvariableopHsavev2_adam_recommender_net_embedding_3_embeddings_m_read_readvariableopFsavev2_adam_recommender_net_embedding_embeddings_v_read_readvariableopHsavev2_adam_recommender_net_embedding_1_embeddings_v_read_readvariableopHsavev2_adam_recommender_net_embedding_2_embeddings_v_read_readvariableopHsavev2_adam_recommender_net_embedding_3_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�)2:	�):	�2:	�: : : : : : : :	�)2:	�):	�2:	�:	�)2:	�):	�2:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�)2:%!

_output_shapes
:	�):%!

_output_shapes
:	�2:%!

_output_shapes
:	�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�)2:%!

_output_shapes
:	�):%!

_output_shapes
:	�2:%!

_output_shapes
:	�:%!

_output_shapes
:	�)2:%!

_output_shapes
:	�):%!

_output_shapes
:	�2:%!

_output_shapes
:	�:

_output_shapes
: 
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_12665

inputs	)
embedding_lookup_12655:	�)2
identity��embedding_lookup�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_12655inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12655*'
_output_shapes
:���������2*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12655*'
_output_shapes
:���������2}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_12655*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^embedding_lookupG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_embedding_3_layer_call_fn_12708

inputs	
unknown:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_12303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_embedding_2_layer_call_fn_12688

inputs	
unknown:	�2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_12286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_embedding_1_layer_call_fn_12672

inputs	
unknown:	�)
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_12265o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_12645d
Qrecommender_net_embedding_2_embeddings_regularizer_l2loss_readvariableop_resource:	�2
identity��Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpQrecommender_net_embedding_2_embeddings_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity:recommender_net/embedding_2/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp
�
�
F__inference_embedding_2_layer_call_and_return_conditional_losses_12701

inputs	)
embedding_lookup_12691:	�2
identity��embedding_lookup�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_12691inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12691*'
_output_shapes
:���������2*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12691*'
_output_shapes
:���������2}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_12691*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^embedding_lookupI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_embedding_3_layer_call_and_return_conditional_losses_12717

inputs	)
embedding_lookup_12711:	�
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_12711inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12711*'
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12711*'
_output_shapes
:���������}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_12681

inputs	)
embedding_lookup_12675:	�)
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_12675inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12675*'
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12675*'
_output_shapes
:���������}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
)__inference_embedding_layer_call_fn_12652

inputs	
unknown:	�)2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12248o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_recommender_net_layer_call_fn_12366
input_1	
unknown:	�)2
	unknown_0:	�)
	unknown_1:	�2
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_recommender_net_layer_call_and_return_conditional_losses_12355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
F__inference_embedding_1_layer_call_and_return_conditional_losses_12265

inputs	)
embedding_lookup_12259:	�)
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_12259inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12259*'
_output_shapes
:���������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12259*'
_output_shapes
:���������}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�\
�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12490
input_1	"
embedding_12418:	�)2$
embedding_1_12425:	�)$
embedding_2_12432:	�2$
embedding_3_12439:	�
identity��!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_12418*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_12248f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_12425*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_12265f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSliceinput_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_12432*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_12286f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_3StridedSliceinput_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_12439*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_12303_
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       Q
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB i
Tensordot/ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	Transpose*embedding/StatefulPartitionedCall:output:0Tensordot/concat:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������a
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB m
Tensordot/Shape_1Shape,embedding_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:[
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1:output:0Tensordot/free_1:output:0"Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1:output:0Tensordot/axes_1:output:0"Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_2ProdTensordot/GatherV2_2:output:0Tensordot/Const_2:output:0*
T0*
_output_shapes
: [
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_3ProdTensordot/GatherV2_3:output:0Tensordot/Const_3:output:0*
T0*
_output_shapes
: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/axes_1:output:0Tensordot/free_1:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
Tensordot/stack_1PackTensordot/Prod_3:output:0Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose_1	Transpose,embedding_2/StatefulPartitionedCall:output:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������Y
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_2ConcatV2Tensordot/GatherV2:output:0Tensordot/GatherV2_2:output:0 Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: n
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
addAddV2Tensordot:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������w
add_1AddV2add:z:0,embedding_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_12418*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_2_12432*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
#__inference_signature_wrapper_12519
input_1	
unknown:	�)2
	unknown_0:	�)
	unknown_1:	�2
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_12224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
D__inference_embedding_layer_call_and_return_conditional_losses_12248

inputs	)
embedding_lookup_12238:	�)2
identity��embedding_lookup�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_12238inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12238*'
_output_shapes
:���������2*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12238*'
_output_shapes
:���������2}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_12238*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^embedding_lookupG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�b
�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12619

inputs	3
 embedding_embedding_lookup_12539:	�)25
"embedding_1_embedding_lookup_12548:	�)5
"embedding_2_embedding_lookup_12557:	�25
"embedding_3_embedding_lookup_12566:	�
identity��embedding/embedding_lookup�embedding_1/embedding_lookup�embedding_2/embedding_lookup�embedding_3/embedding_lookup�Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
embedding/embedding_lookupResourceGather embedding_embedding_lookup_12539strided_slice:output:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/12539*'
_output_shapes
:���������2*
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/12539*'
_output_shapes
:���������2�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_12548strided_slice_1:output:0*
Tindices0	*5
_class+
)'loc:@embedding_1/embedding_lookup/12548*'
_output_shapes
:���������*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/12548*'
_output_shapes
:����������
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_12557strided_slice_2:output:0*
Tindices0	*5
_class+
)'loc:@embedding_2/embedding_lookup/12557*'
_output_shapes
:���������2*
dtype0�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/12557*'
_output_shapes
:���������2�
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
embedding_3/embedding_lookupResourceGather"embedding_3_embedding_lookup_12566strided_slice_3:output:0*
Tindices0	*5
_class+
)'loc:@embedding_3/embedding_lookup/12566*'
_output_shapes
:���������*
dtype0�
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_3/embedding_lookup/12566*'
_output_shapes
:����������
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������_
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       Q
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB m
Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0Tensordot/concat:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������a
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB q
Tensordot/Shape_1Shape0embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:[
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1:output:0Tensordot/free_1:output:0"Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: [
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1:output:0Tensordot/axes_1:output:0"Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_2ProdTensordot/GatherV2_2:output:0Tensordot/Const_2:output:0*
T0*
_output_shapes
: [
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_3ProdTensordot/GatherV2_3:output:0Tensordot/Const_3:output:0*
T0*
_output_shapes
: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/axes_1:output:0Tensordot/free_1:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:}
Tensordot/stack_1PackTensordot/Prod_3:output:0Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose_1	Transpose0embedding_2/embedding_lookup/Identity_1:output:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:���������2�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������Y
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_2ConcatV2Tensordot/GatherV2:output:0Tensordot/GatherV2_2:output:0 Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: n
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
addAddV2Tensordot:output:00embedding_1/embedding_lookup/Identity_1:output:0*
T0*'
_output_shapes
:���������{
add_1AddV2add:z:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*'
_output_shapes
:���������O
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:����������
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOp embedding_embedding_lookup_12539*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOp"embedding_2_embedding_lookup_12557*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^embedding/embedding_lookup^embedding_1/embedding_lookup^embedding_2/embedding_lookup^embedding_3/embedding_lookupG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_embedding_2_layer_call_and_return_conditional_losses_12286

inputs	)
embedding_lookup_12276:	�2
identity��embedding_lookup�Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp�
embedding_lookupResourceGatherembedding_lookup_12276inputs*
Tindices0	*)
_class
loc:@embedding_lookup/12276*'
_output_shapes
:���������2*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12276*'
_output_shapes
:���������2}
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpembedding_lookup_12276*
_output_shapes
:	�2*
dtype0�
9recommender_net/embedding_2/embeddings/Regularizer/L2LossL2LossPrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: }
8recommender_net/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
6recommender_net/embedding_2/embeddings/Regularizer/mulMulArecommender_net/embedding_2/embeddings/Regularizer/mul/x:output:0Brecommender_net/embedding_2/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: s
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:���������2�
NoOpNoOp^embedding_lookupI^recommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup2�
Hrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOpHrecommender_net/embedding_2/embeddings/Regularizer/L2Loss/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_12636b
Orecommender_net_embedding_embeddings_regularizer_l2loss_readvariableop_resource:	�)2
identity��Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpReadVariableOpOrecommender_net_embedding_embeddings_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�)2*
dtype0�
7recommender_net/embedding/embeddings/Regularizer/L2LossL2LossNrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: {
6recommender_net/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�76�
4recommender_net/embedding/embeddings/Regularizer/mulMul?recommender_net/embedding/embeddings/Regularizer/mul/x:output:0@recommender_net/embedding/embeddings/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: v
IdentityIdentity8recommender_net/embedding/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOpG^recommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
Frecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOpFrecommender_net/embedding/embeddings/Regularizer/L2Loss/ReadVariableOp
�h
�
 __inference__wrapped_model_12224
input_1	C
0recommender_net_embedding_embedding_lookup_12152:	�)2E
2recommender_net_embedding_1_embedding_lookup_12161:	�)E
2recommender_net_embedding_2_embedding_lookup_12170:	�2E
2recommender_net_embedding_3_embedding_lookup_12179:	�
identity��*recommender_net/embedding/embedding_lookup�,recommender_net/embedding_1/embedding_lookup�,recommender_net/embedding_2/embedding_lookup�,recommender_net/embedding_3/embedding_lookupt
#recommender_net/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%recommender_net/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%recommender_net/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_sliceStridedSliceinput_1,recommender_net/strided_slice/stack:output:0.recommender_net/strided_slice/stack_1:output:0.recommender_net/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
*recommender_net/embedding/embedding_lookupResourceGather0recommender_net_embedding_embedding_lookup_12152&recommender_net/strided_slice:output:0*
Tindices0	*C
_class9
75loc:@recommender_net/embedding/embedding_lookup/12152*'
_output_shapes
:���������2*
dtype0�
3recommender_net/embedding/embedding_lookup/IdentityIdentity3recommender_net/embedding/embedding_lookup:output:0*
T0*C
_class9
75loc:@recommender_net/embedding/embedding_lookup/12152*'
_output_shapes
:���������2�
5recommender_net/embedding/embedding_lookup/Identity_1Identity<recommender_net/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2v
%recommender_net/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'recommender_net/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_1StridedSliceinput_1.recommender_net/strided_slice_1/stack:output:00recommender_net/strided_slice_1/stack_1:output:00recommender_net/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_1/embedding_lookupResourceGather2recommender_net_embedding_1_embedding_lookup_12161(recommender_net/strided_slice_1:output:0*
Tindices0	*E
_class;
97loc:@recommender_net/embedding_1/embedding_lookup/12161*'
_output_shapes
:���������*
dtype0�
5recommender_net/embedding_1/embedding_lookup/IdentityIdentity5recommender_net/embedding_1/embedding_lookup:output:0*
T0*E
_class;
97loc:@recommender_net/embedding_1/embedding_lookup/12161*'
_output_shapes
:����������
7recommender_net/embedding_1/embedding_lookup/Identity_1Identity>recommender_net/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������v
%recommender_net/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_2StridedSliceinput_1.recommender_net/strided_slice_2/stack:output:00recommender_net/strided_slice_2/stack_1:output:00recommender_net/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_2/embedding_lookupResourceGather2recommender_net_embedding_2_embedding_lookup_12170(recommender_net/strided_slice_2:output:0*
Tindices0	*E
_class;
97loc:@recommender_net/embedding_2/embedding_lookup/12170*'
_output_shapes
:���������2*
dtype0�
5recommender_net/embedding_2/embedding_lookup/IdentityIdentity5recommender_net/embedding_2/embedding_lookup:output:0*
T0*E
_class;
97loc:@recommender_net/embedding_2/embedding_lookup/12170*'
_output_shapes
:���������2�
7recommender_net/embedding_2/embedding_lookup/Identity_1Identity>recommender_net/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������2v
%recommender_net/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'recommender_net/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
recommender_net/strided_slice_3StridedSliceinput_1.recommender_net/strided_slice_3/stack:output:00recommender_net/strided_slice_3/stack_1:output:00recommender_net/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
,recommender_net/embedding_3/embedding_lookupResourceGather2recommender_net_embedding_3_embedding_lookup_12179(recommender_net/strided_slice_3:output:0*
Tindices0	*E
_class;
97loc:@recommender_net/embedding_3/embedding_lookup/12179*'
_output_shapes
:���������*
dtype0�
5recommender_net/embedding_3/embedding_lookup/IdentityIdentity5recommender_net/embedding_3/embedding_lookup:output:0*
T0*E
_class;
97loc:@recommender_net/embedding_3/embedding_lookup/12179*'
_output_shapes
:����������
7recommender_net/embedding_3/embedding_lookup/Identity_1Identity>recommender_net/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:���������o
recommender_net/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       a
recommender_net/Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB �
recommender_net/Tensordot/ShapeShape>recommender_net/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:i
'recommender_net/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/GatherV2GatherV2(recommender_net/Tensordot/Shape:output:0'recommender_net/Tensordot/free:output:00recommender_net/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: k
)recommender_net/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_1GatherV2(recommender_net/Tensordot/Shape:output:0'recommender_net/Tensordot/axes:output:02recommender_net/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
recommender_net/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
recommender_net/Tensordot/ProdProd+recommender_net/Tensordot/GatherV2:output:0(recommender_net/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!recommender_net/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_1Prod-recommender_net/Tensordot/GatherV2_1:output:0*recommender_net/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%recommender_net/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 recommender_net/Tensordot/concatConcatV2'recommender_net/Tensordot/free:output:0'recommender_net/Tensordot/axes:output:0.recommender_net/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
recommender_net/Tensordot/stackPack'recommender_net/Tensordot/Prod:output:0)recommender_net/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
#recommender_net/Tensordot/transpose	Transpose>recommender_net/embedding/embedding_lookup/Identity_1:output:0)recommender_net/Tensordot/concat:output:0*
T0*'
_output_shapes
:���������2�
!recommender_net/Tensordot/ReshapeReshape'recommender_net/Tensordot/transpose:y:0(recommender_net/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������q
 recommender_net/Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       c
 recommender_net/Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB �
!recommender_net/Tensordot/Shape_1Shape@recommender_net/embedding_2/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:k
)recommender_net/Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_2GatherV2*recommender_net/Tensordot/Shape_1:output:0)recommender_net/Tensordot/free_1:output:02recommender_net/Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: k
)recommender_net/Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$recommender_net/Tensordot/GatherV2_3GatherV2*recommender_net/Tensordot/Shape_1:output:0)recommender_net/Tensordot/axes_1:output:02recommender_net/Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!recommender_net/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_2Prod-recommender_net/Tensordot/GatherV2_2:output:0*recommender_net/Tensordot/Const_2:output:0*
T0*
_output_shapes
: k
!recommender_net/Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: �
 recommender_net/Tensordot/Prod_3Prod-recommender_net/Tensordot/GatherV2_3:output:0*recommender_net/Tensordot/Const_3:output:0*
T0*
_output_shapes
: i
'recommender_net/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/concat_1ConcatV2)recommender_net/Tensordot/axes_1:output:0)recommender_net/Tensordot/free_1:output:00recommender_net/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
!recommender_net/Tensordot/stack_1Pack)recommender_net/Tensordot/Prod_3:output:0)recommender_net/Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:�
%recommender_net/Tensordot/transpose_1	Transpose@recommender_net/embedding_2/embedding_lookup/Identity_1:output:0+recommender_net/Tensordot/concat_1:output:0*
T0*'
_output_shapes
:���������2�
#recommender_net/Tensordot/Reshape_1Reshape)recommender_net/Tensordot/transpose_1:y:0*recommender_net/Tensordot/stack_1:output:0*
T0*0
_output_shapes
:�������������������
 recommender_net/Tensordot/MatMulMatMul*recommender_net/Tensordot/Reshape:output:0,recommender_net/Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:������������������i
'recommender_net/Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"recommender_net/Tensordot/concat_2ConcatV2+recommender_net/Tensordot/GatherV2:output:0-recommender_net/Tensordot/GatherV2_2:output:00recommender_net/Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: �
recommender_net/TensordotReshape*recommender_net/Tensordot/MatMul:product:0+recommender_net/Tensordot/concat_2:output:0*
T0*
_output_shapes
: �
recommender_net/addAddV2"recommender_net/Tensordot:output:0@recommender_net/embedding_1/embedding_lookup/Identity_1:output:0*
T0*'
_output_shapes
:����������
recommender_net/add_1AddV2recommender_net/add:z:0@recommender_net/embedding_3/embedding_lookup/Identity_1:output:0*
T0*'
_output_shapes
:���������o
recommender_net/SigmoidSigmoidrecommender_net/add_1:z:0*
T0*'
_output_shapes
:���������j
IdentityIdentityrecommender_net/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp+^recommender_net/embedding/embedding_lookup-^recommender_net/embedding_1/embedding_lookup-^recommender_net/embedding_2/embedding_lookup-^recommender_net/embedding_3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2X
*recommender_net/embedding/embedding_lookup*recommender_net/embedding/embedding_lookup2\
,recommender_net/embedding_1/embedding_lookup,recommender_net/embedding_1/embedding_lookup2\
,recommender_net/embedding_2/embedding_lookup,recommender_net/embedding_2/embedding_lookup2\
,recommender_net/embedding_3/embedding_lookup,recommender_net/embedding_3/embedding_lookup:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
/__inference_recommender_net_layer_call_fn_12532

inputs	
unknown:	�)2
	unknown_0:	�)
	unknown_1:	�2
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_recommender_net_layer_call_and_return_conditional_losses_12355o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Q
�
!__inference__traced_restore_12864
file_prefixH
5assignvariableop_recommender_net_embedding_embeddings:	�)2L
9assignvariableop_1_recommender_net_embedding_1_embeddings:	�)L
9assignvariableop_2_recommender_net_embedding_2_embeddings:	�2L
9assignvariableop_3_recommender_net_embedding_3_embeddings:	�&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: R
?assignvariableop_11_adam_recommender_net_embedding_embeddings_m:	�)2T
Aassignvariableop_12_adam_recommender_net_embedding_1_embeddings_m:	�)T
Aassignvariableop_13_adam_recommender_net_embedding_2_embeddings_m:	�2T
Aassignvariableop_14_adam_recommender_net_embedding_3_embeddings_m:	�R
?assignvariableop_15_adam_recommender_net_embedding_embeddings_v:	�)2T
Aassignvariableop_16_adam_recommender_net_embedding_1_embeddings_v:	�)T
Aassignvariableop_17_adam_recommender_net_embedding_2_embeddings_v:	�2T
Aassignvariableop_18_adam_recommender_net_embedding_3_embeddings_v:	�
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp5assignvariableop_recommender_net_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp9assignvariableop_1_recommender_net_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp9assignvariableop_2_recommender_net_embedding_2_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_recommender_net_embedding_3_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp?assignvariableop_11_adam_recommender_net_embedding_embeddings_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpAassignvariableop_12_adam_recommender_net_embedding_1_embeddings_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpAassignvariableop_13_adam_recommender_net_embedding_2_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpAassignvariableop_14_adam_recommender_net_embedding_3_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp?assignvariableop_15_adam_recommender_net_embedding_embeddings_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpAassignvariableop_16_adam_recommender_net_embedding_1_embeddings_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_adam_recommender_net_embedding_2_embeddings_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpAassignvariableop_18_adam_recommender_net_embedding_3_embeddings_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0	���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�p
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
user_embedding
		user_bias

game_embedding
	game_bias
	optimizer

signatures"
_tf_keras_model
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
/__inference_recommender_net_layer_call_fn_12366
/__inference_recommender_net_layer_call_fn_12532�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12619
J__inference_recommender_net_layer_call_and_return_conditional_losses_12490�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
 __inference__wrapped_model_12224input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
5iter

6beta_1

7beta_2
	8decay
9learning_ratem^m_m`mavbvcvdve"
	optimizer
,
:serving_default"
signature_map
7:5	�)22$recommender_net/embedding/embeddings
9:7	�)2&recommender_net/embedding_1/embeddings
9:7	�22&recommender_net/embedding_2/embeddings
9:7	�2&recommender_net/embedding_3/embeddings
�
;trace_02�
__inference_loss_fn_0_12636�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z;trace_0
�
<trace_02�
__inference_loss_fn_1_12645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z<trace_0
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_recommender_net_layer_call_fn_12366input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_recommender_net_layer_call_fn_12532inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12619inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_recommender_net_layer_call_and_return_conditional_losses_12490input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
)__inference_embedding_layer_call_fn_12652�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0
�
Dtrace_02�
D__inference_embedding_layer_call_and_return_conditional_losses_12665�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
+__inference_embedding_1_layer_call_fn_12672�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0
�
Ktrace_02�
F__inference_embedding_1_layer_call_and_return_conditional_losses_12681�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_02�
+__inference_embedding_2_layer_call_fn_12688�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0
�
Rtrace_02�
F__inference_embedding_2_layer_call_and_return_conditional_losses_12701�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_02�
+__inference_embedding_3_layer_call_fn_12708�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0
�
Ytrace_02�
F__inference_embedding_3_layer_call_and_return_conditional_losses_12717�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
#__inference_signature_wrapper_12519input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_12636"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_12645"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
N
Z	variables
[	keras_api
	\total
	]count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_embedding_layer_call_fn_12652inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_embedding_layer_call_and_return_conditional_losses_12665inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_embedding_1_layer_call_fn_12672inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_embedding_1_layer_call_and_return_conditional_losses_12681inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_embedding_2_layer_call_fn_12688inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_embedding_2_layer_call_and_return_conditional_losses_12701inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_embedding_3_layer_call_fn_12708inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_embedding_3_layer_call_and_return_conditional_losses_12717inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
\0
]1"
trackable_list_wrapper
-
Z	variables"
_generic_user_object
:  (2total
:  (2count
<::	�)22+Adam/recommender_net/embedding/embeddings/m
>:<	�)2-Adam/recommender_net/embedding_1/embeddings/m
>:<	�22-Adam/recommender_net/embedding_2/embeddings/m
>:<	�2-Adam/recommender_net/embedding_3/embeddings/m
<::	�)22+Adam/recommender_net/embedding/embeddings/v
>:<	�)2-Adam/recommender_net/embedding_1/embeddings/v
>:<	�22-Adam/recommender_net/embedding_2/embeddings/v
>:<	�2-Adam/recommender_net/embedding_3/embeddings/v�
 __inference__wrapped_model_12224m0�-
&�#
!�
input_1���������	
� "3�0
.
output_1"�
output_1����������
F__inference_embedding_1_layer_call_and_return_conditional_losses_12681W+�(
!�
�
inputs���������	
� "%�"
�
0���������
� y
+__inference_embedding_1_layer_call_fn_12672J+�(
!�
�
inputs���������	
� "�����������
F__inference_embedding_2_layer_call_and_return_conditional_losses_12701W+�(
!�
�
inputs���������	
� "%�"
�
0���������2
� y
+__inference_embedding_2_layer_call_fn_12688J+�(
!�
�
inputs���������	
� "����������2�
F__inference_embedding_3_layer_call_and_return_conditional_losses_12717W+�(
!�
�
inputs���������	
� "%�"
�
0���������
� y
+__inference_embedding_3_layer_call_fn_12708J+�(
!�
�
inputs���������	
� "�����������
D__inference_embedding_layer_call_and_return_conditional_losses_12665W+�(
!�
�
inputs���������	
� "%�"
�
0���������2
� w
)__inference_embedding_layer_call_fn_12652J+�(
!�
�
inputs���������	
� "����������2:
__inference_loss_fn_0_12636�

� 
� "� :
__inference_loss_fn_1_12645�

� 
� "� �
J__inference_recommender_net_layer_call_and_return_conditional_losses_12490_0�-
&�#
!�
input_1���������	
� "%�"
�
0���������
� �
J__inference_recommender_net_layer_call_and_return_conditional_losses_12619^/�,
%�"
 �
inputs���������	
� "%�"
�
0���������
� �
/__inference_recommender_net_layer_call_fn_12366R0�-
&�#
!�
input_1���������	
� "�����������
/__inference_recommender_net_layer_call_fn_12532Q/�,
%�"
 �
inputs���������	
� "�����������
#__inference_signature_wrapper_12519x;�8
� 
1�.
,
input_1!�
input_1���������	"3�0
.
output_1"�
output_1���������