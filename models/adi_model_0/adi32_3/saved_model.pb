он
рЅ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8“Ъ
~
board1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard1/kernel
w
!board1/kernel/Read/ReadVariableOpReadVariableOpboard1/kernel*&
_output_shapes
:*
dtype0
n
board1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard1/bias
g
board1/bias/Read/ReadVariableOpReadVariableOpboard1/bias*
_output_shapes
:*
dtype0
~
board2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard2/kernel
w
!board2/kernel/Read/ReadVariableOpReadVariableOpboard2/kernel*&
_output_shapes
:*
dtype0
n
board2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard2/bias
g
board2/bias/Read/ReadVariableOpReadVariableOpboard2/bias*
_output_shapes
:*
dtype0
В
fileconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefileconv/kernel
{
#fileconv/kernel/Read/ReadVariableOpReadVariableOpfileconv/kernel*&
_output_shapes
:*
dtype0
r
fileconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefileconv/bias
k
!fileconv/bias/Read/ReadVariableOpReadVariableOpfileconv/bias*
_output_shapes
:*
dtype0
В
rankconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namerankconv/kernel
{
#rankconv/kernel/Read/ReadVariableOpReadVariableOprankconv/kernel*&
_output_shapes
:*
dtype0
r
rankconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namerankconv/bias
k
!rankconv/bias/Read/ReadVariableOpReadVariableOprankconv/bias*
_output_shapes
:*
dtype0
И
quarterconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namequarterconv/kernel
Б
&quarterconv/kernel/Read/ReadVariableOpReadVariableOpquarterconv/kernel*&
_output_shapes
:*
dtype0
x
quarterconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namequarterconv/bias
q
$quarterconv/bias/Read/ReadVariableOpReadVariableOpquarterconv/bias*
_output_shapes
:*
dtype0
Д
largeconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namelargeconv/kernel
}
$largeconv/kernel/Read/ReadVariableOpReadVariableOplargeconv/kernel*&
_output_shapes
:*
dtype0
t
largeconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelargeconv/bias
m
"largeconv/bias/Read/ReadVariableOpReadVariableOplargeconv/bias*
_output_shapes
:*
dtype0
~
board3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard3/kernel
w
!board3/kernel/Read/ReadVariableOpReadVariableOpboard3/kernel*&
_output_shapes
:*
dtype0
n
board3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameboard3/bias
g
board3/bias/Read/ReadVariableOpReadVariableOpboard3/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
§А*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
§А*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	А@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@ *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
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

NoOpNoOp
хO
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*∞O
value¶OB£O BЬO
ƒ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
R
\	variables
]trainable_variables
^regularization_losses
_	keras_api
R
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
h

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api
h

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
n
Вkernel
	Гbias
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
 
 
®
0
1
$2
%3
*4
+5
06
17
68
79
<10
=11
B12
C13
h14
i15
n16
o17
x18
y19
В20
Г21
®
0
1
$2
%3
*4
+5
06
17
68
79
<10
=11
B12
C13
h14
i15
n16
o17
x18
y19
В20
Г21
 
≤
	variables
Иmetrics
Йlayer_metrics
trainable_variables
Кnon_trainable_variables
regularization_losses
Лlayers
 Мlayer_regularization_losses
 
YW
VARIABLE_VALUEboard1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEboard1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≤
 	variables
Нmetrics
Оlayer_metrics
!trainable_variables
Пnon_trainable_variables
"regularization_losses
Рlayers
 Сlayer_regularization_losses
YW
VARIABLE_VALUEboard2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEboard2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
≤
&	variables
Тmetrics
Уlayer_metrics
'trainable_variables
Фnon_trainable_variables
(regularization_losses
Хlayers
 Цlayer_regularization_losses
[Y
VARIABLE_VALUEfileconv/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEfileconv/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
≤
,	variables
Чmetrics
Шlayer_metrics
-trainable_variables
Щnon_trainable_variables
.regularization_losses
Ъlayers
 Ыlayer_regularization_losses
[Y
VARIABLE_VALUErankconv/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUErankconv/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
≤
2	variables
Ьmetrics
Эlayer_metrics
3trainable_variables
Юnon_trainable_variables
4regularization_losses
Яlayers
 †layer_regularization_losses
^\
VARIABLE_VALUEquarterconv/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEquarterconv/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
≤
8	variables
°metrics
Ґlayer_metrics
9trainable_variables
£non_trainable_variables
:regularization_losses
§layers
 •layer_regularization_losses
\Z
VARIABLE_VALUElargeconv/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUElargeconv/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
≤
>	variables
¶metrics
Іlayer_metrics
?trainable_variables
®non_trainable_variables
@regularization_losses
©layers
 ™layer_regularization_losses
YW
VARIABLE_VALUEboard3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEboard3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
≤
D	variables
Ђmetrics
ђlayer_metrics
Etrainable_variables
≠non_trainable_variables
Fregularization_losses
Ѓlayers
 ѓlayer_regularization_losses
 
 
 
≤
H	variables
∞metrics
±layer_metrics
Itrainable_variables
≤non_trainable_variables
Jregularization_losses
≥layers
 іlayer_regularization_losses
 
 
 
≤
L	variables
µmetrics
ґlayer_metrics
Mtrainable_variables
Јnon_trainable_variables
Nregularization_losses
Єlayers
 єlayer_regularization_losses
 
 
 
≤
P	variables
Їmetrics
їlayer_metrics
Qtrainable_variables
Љnon_trainable_variables
Rregularization_losses
љlayers
 Њlayer_regularization_losses
 
 
 
≤
T	variables
њmetrics
јlayer_metrics
Utrainable_variables
Ѕnon_trainable_variables
Vregularization_losses
¬layers
 √layer_regularization_losses
 
 
 
≤
X	variables
ƒmetrics
≈layer_metrics
Ytrainable_variables
∆non_trainable_variables
Zregularization_losses
«layers
 »layer_regularization_losses
 
 
 
≤
\	variables
…metrics
 layer_metrics
]trainable_variables
Ћnon_trainable_variables
^regularization_losses
ћlayers
 Ќlayer_regularization_losses
 
 
 
≤
`	variables
ќmetrics
ѕlayer_metrics
atrainable_variables
–non_trainable_variables
bregularization_losses
—layers
 “layer_regularization_losses
 
 
 
≤
d	variables
”metrics
‘layer_metrics
etrainable_variables
’non_trainable_variables
fregularization_losses
÷layers
 „layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
≤
j	variables
Ўmetrics
ўlayer_metrics
ktrainable_variables
Џnon_trainable_variables
lregularization_losses
џlayers
 №layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

n0
o1
 
≤
p	variables
Ёmetrics
ёlayer_metrics
qtrainable_variables
яnon_trainable_variables
rregularization_losses
аlayers
 бlayer_regularization_losses
 
 
 
≤
t	variables
вmetrics
гlayer_metrics
utrainable_variables
дnon_trainable_variables
vregularization_losses
еlayers
 жlayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
 
≤
z	variables
зmetrics
иlayer_metrics
{trainable_variables
йnon_trainable_variables
|regularization_losses
кlayers
 лlayer_regularization_losses
 
 
 
≥
~	variables
мmetrics
нlayer_metrics
trainable_variables
оnon_trainable_variables
Аregularization_losses
пlayers
 рlayer_regularization_losses
[Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

В0
Г1

В0
Г1
 
µ
Д	variables
сmetrics
тlayer_metrics
Еtrainable_variables
уnon_trainable_variables
Жregularization_losses
фlayers
 хlayer_regularization_losses

ц0
 
 
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

чtotal

шcount
щ	variables
ъ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ч0
ш1

щ	variables
И
serving_default_statePlaceholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
∞
StatefulPartitionedCallStatefulPartitionedCallserving_default_stateboard1/kernelboard1/biasboard2/kernelboard2/biasboard3/kernelboard3/biaslargeconv/kernellargeconv/biasquarterconv/kernelquarterconv/biasrankconv/kernelrankconv/biasfileconv/kernelfileconv/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_6037314
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!board1/kernel/Read/ReadVariableOpboard1/bias/Read/ReadVariableOp!board2/kernel/Read/ReadVariableOpboard2/bias/Read/ReadVariableOp#fileconv/kernel/Read/ReadVariableOp!fileconv/bias/Read/ReadVariableOp#rankconv/kernel/Read/ReadVariableOp!rankconv/bias/Read/ReadVariableOp&quarterconv/kernel/Read/ReadVariableOp$quarterconv/bias/Read/ReadVariableOp$largeconv/kernel/Read/ReadVariableOp"largeconv/bias/Read/ReadVariableOp!board3/kernel/Read/ReadVariableOpboard3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_save_6038123
Р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameboard1/kernelboard1/biasboard2/kernelboard2/biasfileconv/kernelfileconv/biasrankconv/kernelrankconv/biasquarterconv/kernelquarterconv/biaslargeconv/kernellargeconv/biasboard3/kernelboard3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__traced_restore_6038205тЙ
ђ
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037994

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
”
G
+__inference_flatten_3_layer_call_fn_6037825

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_60365162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы
Э
(__inference_board2_layer_call_fn_6037681

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board2_layer_call_and_return_conditional_losses_60363952
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
п
Б
H__inference_quarterconv_layer_call_and_return_conditional_losses_6037732

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
G
+__inference_flatten_4_layer_call_fn_6037836

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_60365242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю
щ
)__inference_model_1_layer_call_fn_6037592

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:
§А

unknown_14:	А

unknown_15:	А@

unknown_16:@

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_60366502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_6037831

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
н
€
F__inference_largeconv_layer_call_and_return_conditional_losses_6037752

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_6036532

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
ц
D__inference_dense_1_layer_call_and_return_conditional_losses_6036582

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ы
ш
)__inference_model_1_layer_call_fn_6036697	
state!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:
§А

unknown_14:	А

unknown_15:	А@

unknown_16:@

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_60366502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
ж
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_6037820

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€H   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
УР
—
D__inference_model_1_layer_call_and_return_conditional_losses_6037425

inputs?
%board1_conv2d_readvariableop_resource:4
&board1_biasadd_readvariableop_resource:?
%board2_conv2d_readvariableop_resource:4
&board2_biasadd_readvariableop_resource:?
%board3_conv2d_readvariableop_resource:4
&board3_biasadd_readvariableop_resource:B
(largeconv_conv2d_readvariableop_resource:7
)largeconv_biasadd_readvariableop_resource:D
*quarterconv_conv2d_readvariableop_resource:9
+quarterconv_biasadd_readvariableop_resource:A
'rankconv_conv2d_readvariableop_resource:6
(rankconv_biasadd_readvariableop_resource:A
'fileconv_conv2d_readvariableop_resource:6
(fileconv_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
§А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identityИҐboard1/BiasAdd/ReadVariableOpҐboard1/Conv2D/ReadVariableOpҐboard2/BiasAdd/ReadVariableOpҐboard2/Conv2D/ReadVariableOpҐboard3/BiasAdd/ReadVariableOpҐboard3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐfileconv/BiasAdd/ReadVariableOpҐfileconv/Conv2D/ReadVariableOpҐ largeconv/BiasAdd/ReadVariableOpҐlargeconv/Conv2D/ReadVariableOpҐ"quarterconv/BiasAdd/ReadVariableOpҐ!quarterconv/Conv2D/ReadVariableOpҐrankconv/BiasAdd/ReadVariableOpҐrankconv/Conv2D/ReadVariableOp™
board1/Conv2D/ReadVariableOpReadVariableOp%board1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board1/Conv2D/ReadVariableOpє
board1/Conv2DConv2Dinputs$board1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board1/Conv2D°
board1/BiasAdd/ReadVariableOpReadVariableOp&board1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board1/BiasAdd/ReadVariableOp§
board1/BiasAddBiasAddboard1/Conv2D:output:0%board1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board1/BiasAddu
board1/ReluReluboard1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board1/Relu™
board2/Conv2D/ReadVariableOpReadVariableOp%board2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board2/Conv2D/ReadVariableOpћ
board2/Conv2DConv2Dboard1/Relu:activations:0$board2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board2/Conv2D°
board2/BiasAdd/ReadVariableOpReadVariableOp&board2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board2/BiasAdd/ReadVariableOp§
board2/BiasAddBiasAddboard2/Conv2D:output:0%board2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board2/BiasAddu
board2/ReluReluboard2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board2/Relu™
board3/Conv2D/ReadVariableOpReadVariableOp%board3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board3/Conv2D/ReadVariableOpћ
board3/Conv2DConv2Dboard2/Relu:activations:0$board3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board3/Conv2D°
board3/BiasAdd/ReadVariableOpReadVariableOp&board3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board3/BiasAdd/ReadVariableOp§
board3/BiasAddBiasAddboard3/Conv2D:output:0%board3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board3/BiasAddu
board3/ReluReluboard3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board3/Relu≥
largeconv/Conv2D/ReadVariableOpReadVariableOp(largeconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
largeconv/Conv2D/ReadVariableOp¬
largeconv/Conv2DConv2Dinputs'largeconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
largeconv/Conv2D™
 largeconv/BiasAdd/ReadVariableOpReadVariableOp)largeconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 largeconv/BiasAdd/ReadVariableOp∞
largeconv/BiasAddBiasAddlargeconv/Conv2D:output:0(largeconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
largeconv/BiasAdd~
largeconv/ReluRelulargeconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
largeconv/Reluє
!quarterconv/Conv2D/ReadVariableOpReadVariableOp*quarterconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!quarterconv/Conv2D/ReadVariableOp»
quarterconv/Conv2DConv2Dinputs)quarterconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
quarterconv/Conv2D∞
"quarterconv/BiasAdd/ReadVariableOpReadVariableOp+quarterconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"quarterconv/BiasAdd/ReadVariableOpЄ
quarterconv/BiasAddBiasAddquarterconv/Conv2D:output:0*quarterconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
quarterconv/BiasAddД
quarterconv/ReluReluquarterconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
quarterconv/Relu∞
rankconv/Conv2D/ReadVariableOpReadVariableOp'rankconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
rankconv/Conv2D/ReadVariableOpњ
rankconv/Conv2DConv2Dinputs&rankconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
rankconv/Conv2DІ
rankconv/BiasAdd/ReadVariableOpReadVariableOp(rankconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
rankconv/BiasAdd/ReadVariableOpђ
rankconv/BiasAddBiasAddrankconv/Conv2D:output:0'rankconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
rankconv/BiasAdd{
rankconv/ReluRelurankconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
rankconv/Relu∞
fileconv/Conv2D/ReadVariableOpReadVariableOp'fileconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
fileconv/Conv2D/ReadVariableOpњ
fileconv/Conv2DConv2Dinputs&fileconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
fileconv/Conv2DІ
fileconv/BiasAdd/ReadVariableOpReadVariableOp(fileconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
fileconv/BiasAdd/ReadVariableOpђ
fileconv/BiasAddBiasAddfileconv/Conv2D:output:0'fileconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
fileconv/BiasAdd{
fileconv/ReluRelufileconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
fileconv/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstФ
flatten/ReshapeReshapefileconv/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/ConstЪ
flatten_1/ReshapeReshaperankconv/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_2/ConstЭ
flatten_2/ReshapeReshapequarterconv/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€H   2
flatten_3/ConstЫ
flatten_3/ReshapeReshapelargeconv/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_4/ConstЩ
flatten_4/ReshapeReshapeboard1/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`   2
flatten_5/ConstШ
flatten_5/ReshapeReshapeboard3/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2
flatten_5/Reshaper
dense_bass/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
dense_bass/concat/axisµ
dense_bass/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0dense_bass/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€§2
dense_bass/concat
dropout/IdentityIdentitydense_bass/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/Identity°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
§А*
dtype02
dense/MatMul/ReadVariableOpЩ
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/Sigmoid¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_1/MatMul/ReadVariableOpЦ
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/Sigmoidw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_1/dropout/ConstЮ
dropout_1/dropout/MulMuldense_1/Sigmoid:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Mulu
dropout_1/dropout/ShapeShapedense_1/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape“
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_1/dropout/GreaterEqual/yж
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
dropout_1/dropout/GreaterEqualЭ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/CastҐ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Mul_1•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/Sigmoidw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_2/dropout/ConstЮ
dropout_2/dropout/MulMuldense_2/Sigmoid:y:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/Mulu
dropout_2/dropout/ShapeShapedense_2/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape“
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_2/dropout/GreaterEqual/yж
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/CastҐ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/Mul_1•
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp†
dense_3/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЯ
NoOpNoOp^board1/BiasAdd/ReadVariableOp^board1/Conv2D/ReadVariableOp^board2/BiasAdd/ReadVariableOp^board2/Conv2D/ReadVariableOp^board3/BiasAdd/ReadVariableOp^board3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp ^fileconv/BiasAdd/ReadVariableOp^fileconv/Conv2D/ReadVariableOp!^largeconv/BiasAdd/ReadVariableOp ^largeconv/Conv2D/ReadVariableOp#^quarterconv/BiasAdd/ReadVariableOp"^quarterconv/Conv2D/ReadVariableOp ^rankconv/BiasAdd/ReadVariableOp^rankconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2>
board1/BiasAdd/ReadVariableOpboard1/BiasAdd/ReadVariableOp2<
board1/Conv2D/ReadVariableOpboard1/Conv2D/ReadVariableOp2>
board2/BiasAdd/ReadVariableOpboard2/BiasAdd/ReadVariableOp2<
board2/Conv2D/ReadVariableOpboard2/Conv2D/ReadVariableOp2>
board3/BiasAdd/ReadVariableOpboard3/BiasAdd/ReadVariableOp2<
board3/Conv2D/ReadVariableOpboard3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2B
fileconv/BiasAdd/ReadVariableOpfileconv/BiasAdd/ReadVariableOp2@
fileconv/Conv2D/ReadVariableOpfileconv/Conv2D/ReadVariableOp2D
 largeconv/BiasAdd/ReadVariableOp largeconv/BiasAdd/ReadVariableOp2B
largeconv/Conv2D/ReadVariableOplargeconv/Conv2D/ReadVariableOp2H
"quarterconv/BiasAdd/ReadVariableOp"quarterconv/BiasAdd/ReadVariableOp2F
!quarterconv/Conv2D/ReadVariableOp!quarterconv/Conv2D/ReadVariableOp2B
rankconv/BiasAdd/ReadVariableOprankconv/BiasAdd/ReadVariableOp2@
rankconv/Conv2D/ReadVariableOprankconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
ф
%__inference_signature_wrapper_6037314	
state!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:
§А

unknown_14:	А

unknown_15:	А@

unknown_16:@

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_60363602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
ЧЩ
—
D__inference_model_1_layer_call_and_return_conditional_losses_6037543

inputs?
%board1_conv2d_readvariableop_resource:4
&board1_biasadd_readvariableop_resource:?
%board2_conv2d_readvariableop_resource:4
&board2_biasadd_readvariableop_resource:?
%board3_conv2d_readvariableop_resource:4
&board3_biasadd_readvariableop_resource:B
(largeconv_conv2d_readvariableop_resource:7
)largeconv_biasadd_readvariableop_resource:D
*quarterconv_conv2d_readvariableop_resource:9
+quarterconv_biasadd_readvariableop_resource:A
'rankconv_conv2d_readvariableop_resource:6
(rankconv_biasadd_readvariableop_resource:A
'fileconv_conv2d_readvariableop_resource:6
(fileconv_biasadd_readvariableop_resource:8
$dense_matmul_readvariableop_resource:
§А4
%dense_biasadd_readvariableop_resource:	А9
&dense_1_matmul_readvariableop_resource:	А@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identityИҐboard1/BiasAdd/ReadVariableOpҐboard1/Conv2D/ReadVariableOpҐboard2/BiasAdd/ReadVariableOpҐboard2/Conv2D/ReadVariableOpҐboard3/BiasAdd/ReadVariableOpҐboard3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐfileconv/BiasAdd/ReadVariableOpҐfileconv/Conv2D/ReadVariableOpҐ largeconv/BiasAdd/ReadVariableOpҐlargeconv/Conv2D/ReadVariableOpҐ"quarterconv/BiasAdd/ReadVariableOpҐ!quarterconv/Conv2D/ReadVariableOpҐrankconv/BiasAdd/ReadVariableOpҐrankconv/Conv2D/ReadVariableOp™
board1/Conv2D/ReadVariableOpReadVariableOp%board1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board1/Conv2D/ReadVariableOpє
board1/Conv2DConv2Dinputs$board1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board1/Conv2D°
board1/BiasAdd/ReadVariableOpReadVariableOp&board1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board1/BiasAdd/ReadVariableOp§
board1/BiasAddBiasAddboard1/Conv2D:output:0%board1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board1/BiasAddu
board1/ReluReluboard1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board1/Relu™
board2/Conv2D/ReadVariableOpReadVariableOp%board2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board2/Conv2D/ReadVariableOpћ
board2/Conv2DConv2Dboard1/Relu:activations:0$board2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board2/Conv2D°
board2/BiasAdd/ReadVariableOpReadVariableOp&board2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board2/BiasAdd/ReadVariableOp§
board2/BiasAddBiasAddboard2/Conv2D:output:0%board2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board2/BiasAddu
board2/ReluReluboard2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board2/Relu™
board3/Conv2D/ReadVariableOpReadVariableOp%board3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
board3/Conv2D/ReadVariableOpћ
board3/Conv2DConv2Dboard2/Relu:activations:0$board3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
board3/Conv2D°
board3/BiasAdd/ReadVariableOpReadVariableOp&board3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
board3/BiasAdd/ReadVariableOp§
board3/BiasAddBiasAddboard3/Conv2D:output:0%board3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
board3/BiasAddu
board3/ReluReluboard3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
board3/Relu≥
largeconv/Conv2D/ReadVariableOpReadVariableOp(largeconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
largeconv/Conv2D/ReadVariableOp¬
largeconv/Conv2DConv2Dinputs'largeconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
largeconv/Conv2D™
 largeconv/BiasAdd/ReadVariableOpReadVariableOp)largeconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 largeconv/BiasAdd/ReadVariableOp∞
largeconv/BiasAddBiasAddlargeconv/Conv2D:output:0(largeconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
largeconv/BiasAdd~
largeconv/ReluRelulargeconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
largeconv/Reluє
!quarterconv/Conv2D/ReadVariableOpReadVariableOp*quarterconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02#
!quarterconv/Conv2D/ReadVariableOp»
quarterconv/Conv2DConv2Dinputs)quarterconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
quarterconv/Conv2D∞
"quarterconv/BiasAdd/ReadVariableOpReadVariableOp+quarterconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"quarterconv/BiasAdd/ReadVariableOpЄ
quarterconv/BiasAddBiasAddquarterconv/Conv2D:output:0*quarterconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
quarterconv/BiasAddД
quarterconv/ReluReluquarterconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
quarterconv/Relu∞
rankconv/Conv2D/ReadVariableOpReadVariableOp'rankconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
rankconv/Conv2D/ReadVariableOpњ
rankconv/Conv2DConv2Dinputs&rankconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
rankconv/Conv2DІ
rankconv/BiasAdd/ReadVariableOpReadVariableOp(rankconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
rankconv/BiasAdd/ReadVariableOpђ
rankconv/BiasAddBiasAddrankconv/Conv2D:output:0'rankconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
rankconv/BiasAdd{
rankconv/ReluRelurankconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
rankconv/Relu∞
fileconv/Conv2D/ReadVariableOpReadVariableOp'fileconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
fileconv/Conv2D/ReadVariableOpњ
fileconv/Conv2DConv2Dinputs&fileconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
fileconv/Conv2DІ
fileconv/BiasAdd/ReadVariableOpReadVariableOp(fileconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
fileconv/BiasAdd/ReadVariableOpђ
fileconv/BiasAddBiasAddfileconv/Conv2D:output:0'fileconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
fileconv/BiasAdd{
fileconv/ReluRelufileconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
fileconv/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstФ
flatten/ReshapeReshapefileconv/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_1/ConstЪ
flatten_1/ReshapeReshaperankconv/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_1/Reshapes
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten_2/ConstЭ
flatten_2/ReshapeReshapequarterconv/Relu:activations:0flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
flatten_2/Reshapes
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€H   2
flatten_3/ConstЫ
flatten_3/ReshapeReshapelargeconv/Relu:activations:0flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2
flatten_3/Reshapes
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_4/ConstЩ
flatten_4/ReshapeReshapeboard1/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten_4/Reshapes
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`   2
flatten_5/ConstШ
flatten_5/ReshapeReshapeboard3/Relu:activations:0flatten_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2
flatten_5/Reshaper
dense_bass/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
dense_bass/concat/axisµ
dense_bass/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0dense_bass/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€§2
dense_bass/concats
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const†
dropout/dropout/MulMuldense_bass/concat:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/dropout/Mulx
dropout/dropout/ShapeShapedense_bass/concat:output:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeЌ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€§*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yя
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/dropout/GreaterEqualШ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€§2
dropout/dropout/CastЫ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/dropout/Mul_1°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
§А*
dtype02
dense/MatMul/ReadVariableOpЩ
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddt
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/Sigmoid¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_1/MatMul/ReadVariableOpЦ
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_1/Sigmoidw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_1/dropout/ConstЮ
dropout_1/dropout/MulMuldense_1/Sigmoid:y:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Mulu
dropout_1/dropout/ShapeShapedense_1/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape“
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_1/dropout/GreaterEqual/yж
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
dropout_1/dropout/GreaterEqualЭ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/CastҐ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Mul_1•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
dense_2/MatMul/ReadVariableOp†
dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_2/Sigmoidw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_2/dropout/ConstЮ
dropout_2/dropout/MulMuldense_2/Sigmoid:y:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/Mulu
dropout_2/dropout/ShapeShapedense_2/Sigmoid:y:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape“
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2"
 dropout_2/dropout/GreaterEqual/yж
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/CastҐ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout_2/dropout/Mul_1•
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp†
dense_3/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/MatMul§
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp°
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityЯ
NoOpNoOp^board1/BiasAdd/ReadVariableOp^board1/Conv2D/ReadVariableOp^board2/BiasAdd/ReadVariableOp^board2/Conv2D/ReadVariableOp^board3/BiasAdd/ReadVariableOp^board3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp ^fileconv/BiasAdd/ReadVariableOp^fileconv/Conv2D/ReadVariableOp!^largeconv/BiasAdd/ReadVariableOp ^largeconv/Conv2D/ReadVariableOp#^quarterconv/BiasAdd/ReadVariableOp"^quarterconv/Conv2D/ReadVariableOp ^rankconv/BiasAdd/ReadVariableOp^rankconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2>
board1/BiasAdd/ReadVariableOpboard1/BiasAdd/ReadVariableOp2<
board1/Conv2D/ReadVariableOpboard1/Conv2D/ReadVariableOp2>
board2/BiasAdd/ReadVariableOpboard2/BiasAdd/ReadVariableOp2<
board2/Conv2D/ReadVariableOpboard2/Conv2D/ReadVariableOp2>
board3/BiasAdd/ReadVariableOpboard3/BiasAdd/ReadVariableOp2<
board3/Conv2D/ReadVariableOpboard3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2B
fileconv/BiasAdd/ReadVariableOpfileconv/BiasAdd/ReadVariableOp2@
fileconv/Conv2D/ReadVariableOpfileconv/Conv2D/ReadVariableOp2D
 largeconv/BiasAdd/ReadVariableOp largeconv/BiasAdd/ReadVariableOp2B
largeconv/Conv2D/ReadVariableOplargeconv/Conv2D/ReadVariableOp2H
"quarterconv/BiasAdd/ReadVariableOp"quarterconv/BiasAdd/ReadVariableOp2F
!quarterconv/Conv2D/ReadVariableOp!quarterconv/Conv2D/ReadVariableOp2B
rankconv/BiasAdd/ReadVariableOprankconv/BiasAdd/ReadVariableOp2@
rankconv/Conv2D/ReadVariableOprankconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
п
Б
H__inference_quarterconv_layer_call_and_return_conditional_losses_6036446

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_6036524

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к
ь
C__inference_board1_layer_call_and_return_conditional_losses_6037652

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_6036508

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Я
Я
*__inference_fileconv_layer_call_fn_6037701

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fileconv_layer_call_and_return_conditional_losses_60364802
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
b
D__inference_dropout_layer_call_and_return_conditional_losses_6037873

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€§2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
ч
Ч
'__inference_dense_layer_call_fn_6037915

inputs
unknown:
§А
	unknown_0:	А
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_60365652
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€§: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
•
b
)__inference_dropout_layer_call_fn_6037895

inputs
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60367892
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€§2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
ћ

Р
,__inference_dense_bass_layer_call_fn_6037868
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityВ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_bass_layer_call_and_return_conditional_losses_60365452
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€H:€€€€€€€€€ј:€€€€€€€€€`:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€H
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:€€€€€€€€€ј
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€`
"
_user_specified_name
inputs/5
у
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037952

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
м
ю
E__inference_rankconv_layer_call_and_return_conditional_losses_6037712

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
b
F__inference_flatten_3_layer_call_and_return_conditional_losses_6036516

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€H   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
†
+__inference_largeconv_layer_call_fn_6037761

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_largeconv_layer_call_and_return_conditional_losses_60364292
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
b
F__inference_flatten_5_layer_call_and_return_conditional_losses_6037842

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
д]
ї

D__inference_model_1_layer_call_and_return_conditional_losses_6037194	
state(
board1_6037128:
board1_6037130:(
board2_6037133:
board2_6037135:(
board3_6037138:
board3_6037140:+
largeconv_6037143:
largeconv_6037145:-
quarterconv_6037148:!
quarterconv_6037150:*
rankconv_6037153:
rankconv_6037155:*
fileconv_6037158:
fileconv_6037160:!
dense_6037171:
§А
dense_6037173:	А"
dense_1_6037176:	А@
dense_1_6037178:@!
dense_2_6037182:@ 
dense_2_6037184: !
dense_3_6037188: 
dense_3_6037190:
identityИҐboard1/StatefulPartitionedCallҐboard2/StatefulPartitionedCallҐboard3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ fileconv/StatefulPartitionedCallҐ!largeconv/StatefulPartitionedCallҐ#quarterconv/StatefulPartitionedCallҐ rankconv/StatefulPartitionedCallЧ
board1/StatefulPartitionedCallStatefulPartitionedCallstateboard1_6037128board1_6037130*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board1_layer_call_and_return_conditional_losses_60363782 
board1/StatefulPartitionedCallє
board2/StatefulPartitionedCallStatefulPartitionedCall'board1/StatefulPartitionedCall:output:0board2_6037133board2_6037135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board2_layer_call_and_return_conditional_losses_60363952 
board2/StatefulPartitionedCallє
board3/StatefulPartitionedCallStatefulPartitionedCall'board2/StatefulPartitionedCall:output:0board3_6037138board3_6037140*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board3_layer_call_and_return_conditional_losses_60364122 
board3/StatefulPartitionedCall¶
!largeconv/StatefulPartitionedCallStatefulPartitionedCallstatelargeconv_6037143largeconv_6037145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_largeconv_layer_call_and_return_conditional_losses_60364292#
!largeconv/StatefulPartitionedCall∞
#quarterconv/StatefulPartitionedCallStatefulPartitionedCallstatequarterconv_6037148quarterconv_6037150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_quarterconv_layer_call_and_return_conditional_losses_60364462%
#quarterconv/StatefulPartitionedCall°
 rankconv/StatefulPartitionedCallStatefulPartitionedCallstaterankconv_6037153rankconv_6037155*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_rankconv_layer_call_and_return_conditional_losses_60364632"
 rankconv/StatefulPartitionedCall°
 fileconv/StatefulPartitionedCallStatefulPartitionedCallstatefileconv_6037158fileconv_6037160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fileconv_layer_call_and_return_conditional_losses_60364802"
 fileconv/StatefulPartitionedCallш
flatten/PartitionedCallPartitionedCall)fileconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_60364922
flatten/PartitionedCallю
flatten_1/PartitionedCallPartitionedCall)rankconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_60365002
flatten_1/PartitionedCallБ
flatten_2/PartitionedCallPartitionedCall,quarterconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_60365082
flatten_2/PartitionedCall€
flatten_3/PartitionedCallPartitionedCall*largeconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_60365162
flatten_3/PartitionedCallэ
flatten_4/PartitionedCallPartitionedCall'board1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_60365242
flatten_4/PartitionedCallь
flatten_5/PartitionedCallPartitionedCall'board3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_60365322
flatten_5/PartitionedCall≤
dense_bass/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_bass_layer_call_and_return_conditional_losses_60365452
dense_bass/PartitionedCallу
dropout/PartitionedCallPartitionedCall#dense_bass/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60365522
dropout/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6037171dense_6037173*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_60365652
dense/StatefulPartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6037176dense_1_6037178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_60365822!
dense_1/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60366002#
!dropout_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6037182dense_2_6037184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_60366132!
dense_2/StatefulPartitionedCallє
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60366312#
!dropout_2/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_6037188dense_3_6037190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_60366432!
dense_3/StatefulPartitionedCallГ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityП
NoOpNoOp^board1/StatefulPartitionedCall^board2/StatefulPartitionedCall^board3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^fileconv/StatefulPartitionedCall"^largeconv/StatefulPartitionedCall$^quarterconv/StatefulPartitionedCall!^rankconv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2@
board1/StatefulPartitionedCallboard1/StatefulPartitionedCall2@
board2/StatefulPartitionedCallboard2/StatefulPartitionedCall2@
board3/StatefulPartitionedCallboard3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 fileconv/StatefulPartitionedCall fileconv/StatefulPartitionedCall2F
!largeconv/StatefulPartitionedCall!largeconv/StatefulPartitionedCall2J
#quarterconv/StatefulPartitionedCall#quarterconv/StatefulPartitionedCall2D
 rankconv/StatefulPartitionedCall rankconv/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
Ѓ®
О
"__inference__wrapped_model_6036360	
stateG
-model_1_board1_conv2d_readvariableop_resource:<
.model_1_board1_biasadd_readvariableop_resource:G
-model_1_board2_conv2d_readvariableop_resource:<
.model_1_board2_biasadd_readvariableop_resource:G
-model_1_board3_conv2d_readvariableop_resource:<
.model_1_board3_biasadd_readvariableop_resource:J
0model_1_largeconv_conv2d_readvariableop_resource:?
1model_1_largeconv_biasadd_readvariableop_resource:L
2model_1_quarterconv_conv2d_readvariableop_resource:A
3model_1_quarterconv_biasadd_readvariableop_resource:I
/model_1_rankconv_conv2d_readvariableop_resource:>
0model_1_rankconv_biasadd_readvariableop_resource:I
/model_1_fileconv_conv2d_readvariableop_resource:>
0model_1_fileconv_biasadd_readvariableop_resource:@
,model_1_dense_matmul_readvariableop_resource:
§А<
-model_1_dense_biasadd_readvariableop_resource:	АA
.model_1_dense_1_matmul_readvariableop_resource:	А@=
/model_1_dense_1_biasadd_readvariableop_resource:@@
.model_1_dense_2_matmul_readvariableop_resource:@ =
/model_1_dense_2_biasadd_readvariableop_resource: @
.model_1_dense_3_matmul_readvariableop_resource: =
/model_1_dense_3_biasadd_readvariableop_resource:
identityИҐ%model_1/board1/BiasAdd/ReadVariableOpҐ$model_1/board1/Conv2D/ReadVariableOpҐ%model_1/board2/BiasAdd/ReadVariableOpҐ$model_1/board2/Conv2D/ReadVariableOpҐ%model_1/board3/BiasAdd/ReadVariableOpҐ$model_1/board3/Conv2D/ReadVariableOpҐ$model_1/dense/BiasAdd/ReadVariableOpҐ#model_1/dense/MatMul/ReadVariableOpҐ&model_1/dense_1/BiasAdd/ReadVariableOpҐ%model_1/dense_1/MatMul/ReadVariableOpҐ&model_1/dense_2/BiasAdd/ReadVariableOpҐ%model_1/dense_2/MatMul/ReadVariableOpҐ&model_1/dense_3/BiasAdd/ReadVariableOpҐ%model_1/dense_3/MatMul/ReadVariableOpҐ'model_1/fileconv/BiasAdd/ReadVariableOpҐ&model_1/fileconv/Conv2D/ReadVariableOpҐ(model_1/largeconv/BiasAdd/ReadVariableOpҐ'model_1/largeconv/Conv2D/ReadVariableOpҐ*model_1/quarterconv/BiasAdd/ReadVariableOpҐ)model_1/quarterconv/Conv2D/ReadVariableOpҐ'model_1/rankconv/BiasAdd/ReadVariableOpҐ&model_1/rankconv/Conv2D/ReadVariableOp¬
$model_1/board1/Conv2D/ReadVariableOpReadVariableOp-model_1_board1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model_1/board1/Conv2D/ReadVariableOp–
model_1/board1/Conv2DConv2Dstate,model_1/board1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/board1/Conv2Dє
%model_1/board1/BiasAdd/ReadVariableOpReadVariableOp.model_1_board1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/board1/BiasAdd/ReadVariableOpƒ
model_1/board1/BiasAddBiasAddmodel_1/board1/Conv2D:output:0-model_1/board1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board1/BiasAddН
model_1/board1/ReluRelumodel_1/board1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board1/Relu¬
$model_1/board2/Conv2D/ReadVariableOpReadVariableOp-model_1_board2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model_1/board2/Conv2D/ReadVariableOpм
model_1/board2/Conv2DConv2D!model_1/board1/Relu:activations:0,model_1/board2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/board2/Conv2Dє
%model_1/board2/BiasAdd/ReadVariableOpReadVariableOp.model_1_board2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/board2/BiasAdd/ReadVariableOpƒ
model_1/board2/BiasAddBiasAddmodel_1/board2/Conv2D:output:0-model_1/board2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board2/BiasAddН
model_1/board2/ReluRelumodel_1/board2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board2/Relu¬
$model_1/board3/Conv2D/ReadVariableOpReadVariableOp-model_1_board3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model_1/board3/Conv2D/ReadVariableOpм
model_1/board3/Conv2DConv2D!model_1/board2/Relu:activations:0,model_1/board3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/board3/Conv2Dє
%model_1/board3/BiasAdd/ReadVariableOpReadVariableOp.model_1_board3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/board3/BiasAdd/ReadVariableOpƒ
model_1/board3/BiasAddBiasAddmodel_1/board3/Conv2D:output:0-model_1/board3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board3/BiasAddН
model_1/board3/ReluRelumodel_1/board3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/board3/ReluЋ
'model_1/largeconv/Conv2D/ReadVariableOpReadVariableOp0model_1_largeconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/largeconv/Conv2D/ReadVariableOpў
model_1/largeconv/Conv2DConv2Dstate/model_1/largeconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/largeconv/Conv2D¬
(model_1/largeconv/BiasAdd/ReadVariableOpReadVariableOp1model_1_largeconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/largeconv/BiasAdd/ReadVariableOp–
model_1/largeconv/BiasAddBiasAdd!model_1/largeconv/Conv2D:output:00model_1/largeconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/largeconv/BiasAddЦ
model_1/largeconv/ReluRelu"model_1/largeconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/largeconv/Relu—
)model_1/quarterconv/Conv2D/ReadVariableOpReadVariableOp2model_1_quarterconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)model_1/quarterconv/Conv2D/ReadVariableOpя
model_1/quarterconv/Conv2DConv2Dstate1model_1/quarterconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/quarterconv/Conv2D»
*model_1/quarterconv/BiasAdd/ReadVariableOpReadVariableOp3model_1_quarterconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_1/quarterconv/BiasAdd/ReadVariableOpЎ
model_1/quarterconv/BiasAddBiasAdd#model_1/quarterconv/Conv2D:output:02model_1/quarterconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/quarterconv/BiasAddЬ
model_1/quarterconv/ReluRelu$model_1/quarterconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/quarterconv/Relu»
&model_1/rankconv/Conv2D/ReadVariableOpReadVariableOp/model_1_rankconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/rankconv/Conv2D/ReadVariableOp÷
model_1/rankconv/Conv2DConv2Dstate.model_1/rankconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/rankconv/Conv2Dњ
'model_1/rankconv/BiasAdd/ReadVariableOpReadVariableOp0model_1_rankconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/rankconv/BiasAdd/ReadVariableOpћ
model_1/rankconv/BiasAddBiasAdd model_1/rankconv/Conv2D:output:0/model_1/rankconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/rankconv/BiasAddУ
model_1/rankconv/ReluRelu!model_1/rankconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/rankconv/Relu»
&model_1/fileconv/Conv2D/ReadVariableOpReadVariableOp/model_1_fileconv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&model_1/fileconv/Conv2D/ReadVariableOp÷
model_1/fileconv/Conv2DConv2Dstate.model_1/fileconv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
model_1/fileconv/Conv2Dњ
'model_1/fileconv/BiasAdd/ReadVariableOpReadVariableOp0model_1_fileconv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/fileconv/BiasAdd/ReadVariableOpћ
model_1/fileconv/BiasAddBiasAdd model_1/fileconv/Conv2D:output:0/model_1/fileconv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/fileconv/BiasAddУ
model_1/fileconv/ReluRelu!model_1/fileconv/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
model_1/fileconv/Relu
model_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
model_1/flatten/Constі
model_1/flatten/ReshapeReshape#model_1/fileconv/Relu:activations:0model_1/flatten/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/flatten/ReshapeГ
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
model_1/flatten_1/ConstЇ
model_1/flatten_1/ReshapeReshape#model_1/rankconv/Relu:activations:0 model_1/flatten_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/flatten_1/ReshapeГ
model_1/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
model_1/flatten_2/Constљ
model_1/flatten_2/ReshapeReshape&model_1/quarterconv/Relu:activations:0 model_1/flatten_2/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/flatten_2/ReshapeГ
model_1/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€H   2
model_1/flatten_3/Constї
model_1/flatten_3/ReshapeReshape$model_1/largeconv/Relu:activations:0 model_1/flatten_3/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€H2
model_1/flatten_3/ReshapeГ
model_1/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
model_1/flatten_4/Constє
model_1/flatten_4/ReshapeReshape!model_1/board1/Relu:activations:0 model_1/flatten_4/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
model_1/flatten_4/ReshapeГ
model_1/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€`   2
model_1/flatten_5/ConstЄ
model_1/flatten_5/ReshapeReshape!model_1/board3/Relu:activations:0 model_1/flatten_5/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2
model_1/flatten_5/ReshapeВ
model_1/dense_bass/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
model_1/dense_bass/concat/axisэ
model_1/dense_bass/concatConcatV2 model_1/flatten/Reshape:output:0"model_1/flatten_1/Reshape:output:0"model_1/flatten_2/Reshape:output:0"model_1/flatten_3/Reshape:output:0"model_1/flatten_4/Reshape:output:0"model_1/flatten_5/Reshape:output:0'model_1/dense_bass/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€§2
model_1/dense_bass/concatЧ
model_1/dropout/IdentityIdentity"model_1/dense_bass/concat:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
model_1/dropout/Identityє
#model_1/dense/MatMul/ReadVariableOpReadVariableOp,model_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
§А*
dtype02%
#model_1/dense/MatMul/ReadVariableOpє
model_1/dense/MatMulMatMul!model_1/dropout/Identity:output:0+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_1/dense/MatMulЈ
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp-model_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$model_1/dense/BiasAdd/ReadVariableOpЇ
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_1/dense/BiasAddМ
model_1/dense/SigmoidSigmoidmodel_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
model_1/dense/SigmoidЊ
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOpґ
model_1/dense_1/MatMulMatMulmodel_1/dense/Sigmoid:y:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_1/MatMulЉ
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOpЅ
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_1/BiasAddС
model_1/dense_1/SigmoidSigmoid model_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dense_1/SigmoidЗ
model_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2!
model_1/dropout_1/dropout/ConstЊ
model_1/dropout_1/dropout/MulMulmodel_1/dense_1/Sigmoid:y:0(model_1/dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_1/dropout_1/dropout/MulН
model_1/dropout_1/dropout/ShapeShapemodel_1/dense_1/Sigmoid:y:0*
T0*
_output_shapes
:2!
model_1/dropout_1/dropout/Shapeк
6model_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform(model_1/dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype028
6model_1/dropout_1/dropout/random_uniform/RandomUniformЩ
(model_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2*
(model_1/dropout_1/dropout/GreaterEqual/yЖ
&model_1/dropout_1/dropout/GreaterEqualGreaterEqual?model_1/dropout_1/dropout/random_uniform/RandomUniform:output:01model_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&model_1/dropout_1/dropout/GreaterEqualµ
model_1/dropout_1/dropout/CastCast*model_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2 
model_1/dropout_1/dropout/Cast¬
model_1/dropout_1/dropout/Mul_1Mul!model_1/dropout_1/dropout/Mul:z:0"model_1/dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
model_1/dropout_1/dropout/Mul_1љ
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02'
%model_1/dense_2/MatMul/ReadVariableOpј
model_1/dense_2/MatMulMatMul#model_1/dropout_1/dropout/Mul_1:z:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model_1/dense_2/MatMulЉ
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOpЅ
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model_1/dense_2/BiasAddС
model_1/dense_2/SigmoidSigmoid model_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model_1/dense_2/SigmoidЗ
model_1/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2!
model_1/dropout_2/dropout/ConstЊ
model_1/dropout_2/dropout/MulMulmodel_1/dense_2/Sigmoid:y:0(model_1/dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
model_1/dropout_2/dropout/MulН
model_1/dropout_2/dropout/ShapeShapemodel_1/dense_2/Sigmoid:y:0*
T0*
_output_shapes
:2!
model_1/dropout_2/dropout/Shapeк
6model_1/dropout_2/dropout/random_uniform/RandomUniformRandomUniform(model_1/dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype028
6model_1/dropout_2/dropout/random_uniform/RandomUniformЩ
(model_1/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2*
(model_1/dropout_2/dropout/GreaterEqual/yЖ
&model_1/dropout_2/dropout/GreaterEqualGreaterEqual?model_1/dropout_2/dropout/random_uniform/RandomUniform:output:01model_1/dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&model_1/dropout_2/dropout/GreaterEqualµ
model_1/dropout_2/dropout/CastCast*model_1/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2 
model_1/dropout_2/dropout/Cast¬
model_1/dropout_2/dropout/Mul_1Mul!model_1/dropout_2/dropout/Mul:z:0"model_1/dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
model_1/dropout_2/dropout/Mul_1љ
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_3/MatMul/ReadVariableOpј
model_1/dense_3/MatMulMatMul#model_1/dropout_2/dropout/Mul_1:z:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/dense_3/MatMulЉ
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOpЅ
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_1/dense_3/BiasAdd{
IdentityIdentity model_1/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityѕ
NoOpNoOp&^model_1/board1/BiasAdd/ReadVariableOp%^model_1/board1/Conv2D/ReadVariableOp&^model_1/board2/BiasAdd/ReadVariableOp%^model_1/board2/Conv2D/ReadVariableOp&^model_1/board3/BiasAdd/ReadVariableOp%^model_1/board3/Conv2D/ReadVariableOp%^model_1/dense/BiasAdd/ReadVariableOp$^model_1/dense/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp(^model_1/fileconv/BiasAdd/ReadVariableOp'^model_1/fileconv/Conv2D/ReadVariableOp)^model_1/largeconv/BiasAdd/ReadVariableOp(^model_1/largeconv/Conv2D/ReadVariableOp+^model_1/quarterconv/BiasAdd/ReadVariableOp*^model_1/quarterconv/Conv2D/ReadVariableOp(^model_1/rankconv/BiasAdd/ReadVariableOp'^model_1/rankconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2N
%model_1/board1/BiasAdd/ReadVariableOp%model_1/board1/BiasAdd/ReadVariableOp2L
$model_1/board1/Conv2D/ReadVariableOp$model_1/board1/Conv2D/ReadVariableOp2N
%model_1/board2/BiasAdd/ReadVariableOp%model_1/board2/BiasAdd/ReadVariableOp2L
$model_1/board2/Conv2D/ReadVariableOp$model_1/board2/Conv2D/ReadVariableOp2N
%model_1/board3/BiasAdd/ReadVariableOp%model_1/board3/BiasAdd/ReadVariableOp2L
$model_1/board3/Conv2D/ReadVariableOp$model_1/board3/Conv2D/ReadVariableOp2L
$model_1/dense/BiasAdd/ReadVariableOp$model_1/dense/BiasAdd/ReadVariableOp2J
#model_1/dense/MatMul/ReadVariableOp#model_1/dense/MatMul/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2R
'model_1/fileconv/BiasAdd/ReadVariableOp'model_1/fileconv/BiasAdd/ReadVariableOp2P
&model_1/fileconv/Conv2D/ReadVariableOp&model_1/fileconv/Conv2D/ReadVariableOp2T
(model_1/largeconv/BiasAdd/ReadVariableOp(model_1/largeconv/BiasAdd/ReadVariableOp2R
'model_1/largeconv/Conv2D/ReadVariableOp'model_1/largeconv/Conv2D/ReadVariableOp2X
*model_1/quarterconv/BiasAdd/ReadVariableOp*model_1/quarterconv/BiasAdd/ReadVariableOp2V
)model_1/quarterconv/Conv2D/ReadVariableOp)model_1/quarterconv/Conv2D/ReadVariableOp2R
'model_1/rankconv/BiasAdd/ReadVariableOp'model_1/rankconv/BiasAdd/ReadVariableOp2P
&model_1/rankconv/Conv2D/ReadVariableOp&model_1/rankconv/Conv2D/ReadVariableOp:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
Ёf
∞
#__inference__traced_restore_6038205
file_prefix8
assignvariableop_board1_kernel:,
assignvariableop_1_board1_bias::
 assignvariableop_2_board2_kernel:,
assignvariableop_3_board2_bias:<
"assignvariableop_4_fileconv_kernel:.
 assignvariableop_5_fileconv_bias:<
"assignvariableop_6_rankconv_kernel:.
 assignvariableop_7_rankconv_bias:?
%assignvariableop_8_quarterconv_kernel:1
#assignvariableop_9_quarterconv_bias:>
$assignvariableop_10_largeconv_kernel:0
"assignvariableop_11_largeconv_bias:;
!assignvariableop_12_board3_kernel:-
assignvariableop_13_board3_bias:4
 assignvariableop_14_dense_kernel:
§А-
assignvariableop_15_dense_bias:	А5
"assignvariableop_16_dense_1_kernel:	А@.
 assignvariableop_17_dense_1_bias:@4
"assignvariableop_18_dense_2_kernel:@ .
 assignvariableop_19_dense_2_bias: 4
"assignvariableop_20_dense_3_kernel: .
 assignvariableop_21_dense_3_bias:#
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

value—
Bќ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesј
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices®
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_board1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_board1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2•
AssignVariableOp_2AssignVariableOp assignvariableop_2_board2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_board2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp"assignvariableop_4_fileconv_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5•
AssignVariableOp_5AssignVariableOp assignvariableop_5_fileconv_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp"assignvariableop_6_rankconv_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7•
AssignVariableOp_7AssignVariableOp assignvariableop_7_rankconv_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8™
AssignVariableOp_8AssignVariableOp%assignvariableop_8_quarterconv_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9®
AssignVariableOp_9AssignVariableOp#assignvariableop_9_quarterconv_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ђ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_largeconv_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11™
AssignVariableOp_11AssignVariableOp"assignvariableop_11_largeconv_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_board3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_board3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16™
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17®
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18™
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20™
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpо
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24f
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_25÷
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
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
_user_specified_namefile_prefix
д
`
D__inference_flatten_layer_call_and_return_conditional_losses_6036492

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
у
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_6036715

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Я
Я
*__inference_rankconv_layer_call_fn_6037721

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_rankconv_layer_call_and_return_conditional_losses_60364632
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ђ
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_6036631

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€ 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
”
G
+__inference_flatten_5_layer_call_fn_6037847

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_60365322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√
E
)__inference_dropout_layer_call_fn_6037890

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60365522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
Д
х
D__inference_dense_2_layer_call_and_return_conditional_losses_6037973

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
д	
Ђ
G__inference_dense_bass_layer_call_and_return_conditional_losses_6037858
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis™
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€§2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€H:€€€€€€€€€ј:€€€€€€€€€`:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:€€€€€€€€€H
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:€€€€€€€€€ј
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:€€€€€€€€€`
"
_user_specified_name
inputs/5
к
ь
C__inference_board2_layer_call_and_return_conditional_losses_6036395

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶

х
D__inference_dense_3_layer_call_and_return_conditional_losses_6038019

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
д
`
D__inference_flatten_layer_call_and_return_conditional_losses_6037787

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
d
+__inference_dropout_2_layer_call_fn_6038009

inputs
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60366312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ы
Э
(__inference_board3_layer_call_fn_6037781

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board3_layer_call_and_return_conditional_losses_60364122
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
О
ц
B__inference_dense_layer_call_and_return_conditional_losses_6036565

inputs2
matmul_readvariableop_resource:
§А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
§А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Sigmoidg
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€§: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
ж
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_6036500

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
м
ю
E__inference_rankconv_layer_call_and_return_conditional_losses_6036463

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ђ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_6036600

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037999

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
√
G
+__inference_dropout_1_layer_call_fn_6037957

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60367412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
≥
c
D__inference_dropout_layer_call_and_return_conditional_losses_6037885

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€§*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€§2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
ж
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_6037809

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
О
ц
B__inference_dense_layer_call_and_return_conditional_losses_6037906

inputs2
matmul_readvariableop_resource:
§А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
§А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Sigmoidg
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€§: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
х
b
D__inference_dropout_layer_call_and_return_conditional_losses_6036552

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€§2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
л]
Љ

D__inference_model_1_layer_call_and_return_conditional_losses_6036650

inputs(
board1_6036379:
board1_6036381:(
board2_6036396:
board2_6036398:(
board3_6036413:
board3_6036415:+
largeconv_6036430:
largeconv_6036432:-
quarterconv_6036447:!
quarterconv_6036449:*
rankconv_6036464:
rankconv_6036466:*
fileconv_6036481:
fileconv_6036483:!
dense_6036566:
§А
dense_6036568:	А"
dense_1_6036583:	А@
dense_1_6036585:@!
dense_2_6036614:@ 
dense_2_6036616: !
dense_3_6036644: 
dense_3_6036646:
identityИҐboard1/StatefulPartitionedCallҐboard2/StatefulPartitionedCallҐboard3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ fileconv/StatefulPartitionedCallҐ!largeconv/StatefulPartitionedCallҐ#quarterconv/StatefulPartitionedCallҐ rankconv/StatefulPartitionedCallШ
board1/StatefulPartitionedCallStatefulPartitionedCallinputsboard1_6036379board1_6036381*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board1_layer_call_and_return_conditional_losses_60363782 
board1/StatefulPartitionedCallє
board2/StatefulPartitionedCallStatefulPartitionedCall'board1/StatefulPartitionedCall:output:0board2_6036396board2_6036398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board2_layer_call_and_return_conditional_losses_60363952 
board2/StatefulPartitionedCallє
board3/StatefulPartitionedCallStatefulPartitionedCall'board2/StatefulPartitionedCall:output:0board3_6036413board3_6036415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board3_layer_call_and_return_conditional_losses_60364122 
board3/StatefulPartitionedCallІ
!largeconv/StatefulPartitionedCallStatefulPartitionedCallinputslargeconv_6036430largeconv_6036432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_largeconv_layer_call_and_return_conditional_losses_60364292#
!largeconv/StatefulPartitionedCall±
#quarterconv/StatefulPartitionedCallStatefulPartitionedCallinputsquarterconv_6036447quarterconv_6036449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_quarterconv_layer_call_and_return_conditional_losses_60364462%
#quarterconv/StatefulPartitionedCallҐ
 rankconv/StatefulPartitionedCallStatefulPartitionedCallinputsrankconv_6036464rankconv_6036466*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_rankconv_layer_call_and_return_conditional_losses_60364632"
 rankconv/StatefulPartitionedCallҐ
 fileconv/StatefulPartitionedCallStatefulPartitionedCallinputsfileconv_6036481fileconv_6036483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fileconv_layer_call_and_return_conditional_losses_60364802"
 fileconv/StatefulPartitionedCallш
flatten/PartitionedCallPartitionedCall)fileconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_60364922
flatten/PartitionedCallю
flatten_1/PartitionedCallPartitionedCall)rankconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_60365002
flatten_1/PartitionedCallБ
flatten_2/PartitionedCallPartitionedCall,quarterconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_60365082
flatten_2/PartitionedCall€
flatten_3/PartitionedCallPartitionedCall*largeconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_60365162
flatten_3/PartitionedCallэ
flatten_4/PartitionedCallPartitionedCall'board1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_60365242
flatten_4/PartitionedCallь
flatten_5/PartitionedCallPartitionedCall'board3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_60365322
flatten_5/PartitionedCall≤
dense_bass/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_bass_layer_call_and_return_conditional_losses_60365452
dense_bass/PartitionedCallу
dropout/PartitionedCallPartitionedCall#dense_bass/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60365522
dropout/PartitionedCall¶
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6036566dense_6036568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_60365652
dense/StatefulPartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6036583dense_1_6036585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_60365822!
dense_1/StatefulPartitionedCallХ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60366002#
!dropout_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6036614dense_2_6036616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_60366132!
dense_2/StatefulPartitionedCallє
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60366312#
!dropout_2/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_6036644dense_3_6036646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_60366432!
dense_3/StatefulPartitionedCallГ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityП
NoOpNoOp^board1/StatefulPartitionedCall^board2/StatefulPartitionedCall^board3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^fileconv/StatefulPartitionedCall"^largeconv/StatefulPartitionedCall$^quarterconv/StatefulPartitionedCall!^rankconv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2@
board1/StatefulPartitionedCallboard1/StatefulPartitionedCall2@
board2/StatefulPartitionedCallboard2/StatefulPartitionedCall2@
board3/StatefulPartitionedCallboard3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 fileconv/StatefulPartitionedCall fileconv/StatefulPartitionedCall2F
!largeconv/StatefulPartitionedCall!largeconv/StatefulPartitionedCall2J
#quarterconv/StatefulPartitionedCall#quarterconv/StatefulPartitionedCall2D
 rankconv/StatefulPartitionedCall rankconv/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Д
х
D__inference_dense_2_layer_call_and_return_conditional_losses_6036613

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_6036741

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
µ_
ё

D__inference_model_1_layer_call_and_return_conditional_losses_6037029

inputs(
board1_6036963:
board1_6036965:(
board2_6036968:
board2_6036970:(
board3_6036973:
board3_6036975:+
largeconv_6036978:
largeconv_6036980:-
quarterconv_6036983:!
quarterconv_6036985:*
rankconv_6036988:
rankconv_6036990:*
fileconv_6036993:
fileconv_6036995:!
dense_6037006:
§А
dense_6037008:	А"
dense_1_6037011:	А@
dense_1_6037013:@!
dense_2_6037017:@ 
dense_2_6037019: !
dense_3_6037023: 
dense_3_6037025:
identityИҐboard1/StatefulPartitionedCallҐboard2/StatefulPartitionedCallҐboard3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ fileconv/StatefulPartitionedCallҐ!largeconv/StatefulPartitionedCallҐ#quarterconv/StatefulPartitionedCallҐ rankconv/StatefulPartitionedCallШ
board1/StatefulPartitionedCallStatefulPartitionedCallinputsboard1_6036963board1_6036965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board1_layer_call_and_return_conditional_losses_60363782 
board1/StatefulPartitionedCallє
board2/StatefulPartitionedCallStatefulPartitionedCall'board1/StatefulPartitionedCall:output:0board2_6036968board2_6036970*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board2_layer_call_and_return_conditional_losses_60363952 
board2/StatefulPartitionedCallє
board3/StatefulPartitionedCallStatefulPartitionedCall'board2/StatefulPartitionedCall:output:0board3_6036973board3_6036975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board3_layer_call_and_return_conditional_losses_60364122 
board3/StatefulPartitionedCallІ
!largeconv/StatefulPartitionedCallStatefulPartitionedCallinputslargeconv_6036978largeconv_6036980*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_largeconv_layer_call_and_return_conditional_losses_60364292#
!largeconv/StatefulPartitionedCall±
#quarterconv/StatefulPartitionedCallStatefulPartitionedCallinputsquarterconv_6036983quarterconv_6036985*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_quarterconv_layer_call_and_return_conditional_losses_60364462%
#quarterconv/StatefulPartitionedCallҐ
 rankconv/StatefulPartitionedCallStatefulPartitionedCallinputsrankconv_6036988rankconv_6036990*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_rankconv_layer_call_and_return_conditional_losses_60364632"
 rankconv/StatefulPartitionedCallҐ
 fileconv/StatefulPartitionedCallStatefulPartitionedCallinputsfileconv_6036993fileconv_6036995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fileconv_layer_call_and_return_conditional_losses_60364802"
 fileconv/StatefulPartitionedCallш
flatten/PartitionedCallPartitionedCall)fileconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_60364922
flatten/PartitionedCallю
flatten_1/PartitionedCallPartitionedCall)rankconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_60365002
flatten_1/PartitionedCallБ
flatten_2/PartitionedCallPartitionedCall,quarterconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_60365082
flatten_2/PartitionedCall€
flatten_3/PartitionedCallPartitionedCall*largeconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_60365162
flatten_3/PartitionedCallэ
flatten_4/PartitionedCallPartitionedCall'board1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_60365242
flatten_4/PartitionedCallь
flatten_5/PartitionedCallPartitionedCall'board3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_60365322
flatten_5/PartitionedCall≤
dense_bass/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_bass_layer_call_and_return_conditional_losses_60365452
dense_bass/PartitionedCallЛ
dropout/StatefulPartitionedCallStatefulPartitionedCall#dense_bass/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60367892!
dropout/StatefulPartitionedCallЃ
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6037006dense_6037008*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_60365652
dense/StatefulPartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6037011dense_1_6037013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_60365822!
dense_1/StatefulPartitionedCallЈ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60366002#
!dropout_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6037017dense_2_6037019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_60366132!
dense_2/StatefulPartitionedCallє
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60366312#
!dropout_2/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_6037023dense_3_6037025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_60366432!
dense_3/StatefulPartitionedCallГ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp^board1/StatefulPartitionedCall^board2/StatefulPartitionedCall^board3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^fileconv/StatefulPartitionedCall"^largeconv/StatefulPartitionedCall$^quarterconv/StatefulPartitionedCall!^rankconv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2@
board1/StatefulPartitionedCallboard1/StatefulPartitionedCall2@
board2/StatefulPartitionedCallboard2/StatefulPartitionedCall2@
board3/StatefulPartitionedCallboard3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 fileconv/StatefulPartitionedCall fileconv/StatefulPartitionedCall2F
!largeconv/StatefulPartitionedCall!largeconv/StatefulPartitionedCall2J
#quarterconv/StatefulPartitionedCall#quarterconv/StatefulPartitionedCall2D
 rankconv/StatefulPartitionedCall rankconv/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘	
©
G__inference_dense_bass_layer_call_and_return_conditional_losses_6036545

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis®
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€§2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€H:€€€€€€€€€ј:€€€€€€€€€`:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€H
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€`
 
_user_specified_nameinputs
м
ю
E__inference_fileconv_layer_call_and_return_conditional_losses_6037692

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶

х
D__inference_dense_3_layer_call_and_return_conditional_losses_6036643

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
к
ь
C__inference_board1_layer_call_and_return_conditional_losses_6036378

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
ц
D__inference_dense_1_layer_call_and_return_conditional_losses_6037926

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
•
Ґ
-__inference_quarterconv_layer_call_fn_6037741

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_quarterconv_layer_call_and_return_conditional_losses_60364462
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ю
щ
)__inference_model_1_layer_call_fn_6037641

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:
§А

unknown_14:	А

unknown_15:	А@

unknown_16:@

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_60370292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ_
Ё

D__inference_model_1_layer_call_and_return_conditional_losses_6037263	
state(
board1_6037197:
board1_6037199:(
board2_6037202:
board2_6037204:(
board3_6037207:
board3_6037209:+
largeconv_6037212:
largeconv_6037214:-
quarterconv_6037217:!
quarterconv_6037219:*
rankconv_6037222:
rankconv_6037224:*
fileconv_6037227:
fileconv_6037229:!
dense_6037240:
§А
dense_6037242:	А"
dense_1_6037245:	А@
dense_1_6037247:@!
dense_2_6037251:@ 
dense_2_6037253: !
dense_3_6037257: 
dense_3_6037259:
identityИҐboard1/StatefulPartitionedCallҐboard2/StatefulPartitionedCallҐboard3/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ fileconv/StatefulPartitionedCallҐ!largeconv/StatefulPartitionedCallҐ#quarterconv/StatefulPartitionedCallҐ rankconv/StatefulPartitionedCallЧ
board1/StatefulPartitionedCallStatefulPartitionedCallstateboard1_6037197board1_6037199*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board1_layer_call_and_return_conditional_losses_60363782 
board1/StatefulPartitionedCallє
board2/StatefulPartitionedCallStatefulPartitionedCall'board1/StatefulPartitionedCall:output:0board2_6037202board2_6037204*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board2_layer_call_and_return_conditional_losses_60363952 
board2/StatefulPartitionedCallє
board3/StatefulPartitionedCallStatefulPartitionedCall'board2/StatefulPartitionedCall:output:0board3_6037207board3_6037209*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board3_layer_call_and_return_conditional_losses_60364122 
board3/StatefulPartitionedCall¶
!largeconv/StatefulPartitionedCallStatefulPartitionedCallstatelargeconv_6037212largeconv_6037214*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_largeconv_layer_call_and_return_conditional_losses_60364292#
!largeconv/StatefulPartitionedCall∞
#quarterconv/StatefulPartitionedCallStatefulPartitionedCallstatequarterconv_6037217quarterconv_6037219*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_quarterconv_layer_call_and_return_conditional_losses_60364462%
#quarterconv/StatefulPartitionedCall°
 rankconv/StatefulPartitionedCallStatefulPartitionedCallstaterankconv_6037222rankconv_6037224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_rankconv_layer_call_and_return_conditional_losses_60364632"
 rankconv/StatefulPartitionedCall°
 fileconv/StatefulPartitionedCallStatefulPartitionedCallstatefileconv_6037227fileconv_6037229*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_fileconv_layer_call_and_return_conditional_losses_60364802"
 fileconv/StatefulPartitionedCallш
flatten/PartitionedCallPartitionedCall)fileconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_60364922
flatten/PartitionedCallю
flatten_1/PartitionedCallPartitionedCall)rankconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_60365002
flatten_1/PartitionedCallБ
flatten_2/PartitionedCallPartitionedCall,quarterconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_60365082
flatten_2/PartitionedCall€
flatten_3/PartitionedCallPartitionedCall*largeconv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€H* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_3_layer_call_and_return_conditional_losses_60365162
flatten_3/PartitionedCallэ
flatten_4/PartitionedCallPartitionedCall'board1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_60365242
flatten_4/PartitionedCallь
flatten_5/PartitionedCallPartitionedCall'board3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_60365322
flatten_5/PartitionedCall≤
dense_bass/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_dense_bass_layer_call_and_return_conditional_losses_60365452
dense_bass/PartitionedCallЛ
dropout/StatefulPartitionedCallStatefulPartitionedCall#dense_bass/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€§* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_60367892!
dropout/StatefulPartitionedCallЃ
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6037240dense_6037242*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_60365652
dense/StatefulPartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_6037245dense_1_6037247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_60365822!
dense_1/StatefulPartitionedCallЈ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60366002#
!dropout_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_6037251dense_2_6037253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_60366132!
dense_2/StatefulPartitionedCallє
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60366312#
!dropout_2/StatefulPartitionedCallє
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_6037257dense_3_6037259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_60366432!
dense_3/StatefulPartitionedCallГ
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp^board1/StatefulPartitionedCall^board2/StatefulPartitionedCall^board3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall!^fileconv/StatefulPartitionedCall"^largeconv/StatefulPartitionedCall$^quarterconv/StatefulPartitionedCall!^rankconv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 2@
board1/StatefulPartitionedCallboard1/StatefulPartitionedCall2@
board2/StatefulPartitionedCallboard2/StatefulPartitionedCall2@
board3/StatefulPartitionedCallboard3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2D
 fileconv/StatefulPartitionedCall fileconv/StatefulPartitionedCall2F
!largeconv/StatefulPartitionedCall!largeconv/StatefulPartitionedCall2J
#quarterconv/StatefulPartitionedCall#quarterconv/StatefulPartitionedCall2D
 rankconv/StatefulPartitionedCall rankconv/StatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
м
ю
E__inference_fileconv_layer_call_and_return_conditional_losses_6036480

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_6037798

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
•
d
+__inference_dropout_1_layer_call_fn_6037962

inputs
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_60366002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
н
€
F__inference_largeconv_layer_call_and_return_conditional_losses_6036429

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ы
ш
)__inference_model_1_layer_call_fn_6037125	
state!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:

unknown_13:
§А

unknown_14:	А

unknown_15:	А@

unknown_16:@

unknown_17:@ 

unknown_18: 

unknown_19: 

unknown_20:
identityИҐStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_60370292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:€€€€€€€€€

_user_specified_namestate
Ы
Э
(__inference_board1_layer_call_fn_6037661

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_board1_layer_call_and_return_conditional_losses_60363782
StatefulPartitionedCallГ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≥
c
D__inference_dropout_layer_call_and_return_conditional_losses_6036789

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€§*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€§2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€§2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€§2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€§:P L
(
_output_shapes
:€€€€€€€€€§
 
_user_specified_nameinputs
ђ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037947

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
–7
≥	
 __inference__traced_save_6038123
file_prefix,
(savev2_board1_kernel_read_readvariableop*
&savev2_board1_bias_read_readvariableop,
(savev2_board2_kernel_read_readvariableop*
&savev2_board2_bias_read_readvariableop.
*savev2_fileconv_kernel_read_readvariableop,
(savev2_fileconv_bias_read_readvariableop.
*savev2_rankconv_kernel_read_readvariableop,
(savev2_rankconv_bias_read_readvariableop1
-savev2_quarterconv_kernel_read_readvariableop/
+savev2_quarterconv_bias_read_readvariableop/
+savev2_largeconv_kernel_read_readvariableop-
)savev2_largeconv_bias_read_readvariableop,
(savev2_board3_kernel_read_readvariableop*
&savev2_board3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename…
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*џ

value—
Bќ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЄ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_board1_kernel_read_readvariableop&savev2_board1_bias_read_readvariableop(savev2_board2_kernel_read_readvariableop&savev2_board2_bias_read_readvariableop*savev2_fileconv_kernel_read_readvariableop(savev2_fileconv_bias_read_readvariableop*savev2_rankconv_kernel_read_readvariableop(savev2_rankconv_bias_read_readvariableop-savev2_quarterconv_kernel_read_readvariableop+savev2_quarterconv_bias_read_readvariableop+savev2_largeconv_kernel_read_readvariableop)savev2_largeconv_bias_read_readvariableop(savev2_board3_kernel_read_readvariableop&savev2_board3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Й
_input_shapesч
ф: :::::::::::::::
§А:А:	А@:@:@ : : :: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
§А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
Ц
)__inference_dense_2_layer_call_fn_6037982

inputs
unknown:@ 
	unknown_0: 
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_60366132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
к
ь
C__inference_board2_layer_call_and_return_conditional_losses_6037672

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф
Ц
)__inference_dense_3_layer_call_fn_6038028

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_60366432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѕ
E
)__inference_flatten_layer_call_fn_6037792

inputs
identity≈
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_60364922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
√
G
+__inference_dropout_2_layer_call_fn_6038004

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_60367152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€ :O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
к
ь
C__inference_board3_layer_call_and_return_conditional_losses_6037772

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ч
Ч
)__inference_dense_1_layer_call_fn_6037935

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_60365822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
”
G
+__inference_flatten_2_layer_call_fn_6037814

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_60365082
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
”
G
+__inference_flatten_1_layer_call_fn_6037803

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_60365002
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к
ь
C__inference_board3_layer_call_and_return_conditional_losses_6036412

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ѓ
serving_defaultЪ
?
state6
serving_default_state:0€€€€€€€€€;
dense_30
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ѕї
є
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-7
layer-16
layer_with_weights-8
layer-17
layer-18
layer_with_weights-9
layer-19
layer-20
layer_with_weights-10
layer-21
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+ы&call_and_return_all_conditional_losses
ь_default_save_signature
э__call__"
_tf_keras_network
"
_tf_keras_input_layer
љ

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+ю&call_and_return_all_conditional_losses
€__call__"
_tf_keras_layer
љ

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"
_tf_keras_layer
љ

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"
_tf_keras_layer
љ

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layer
љ

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"
_tf_keras_layer
љ

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"
_tf_keras_layer
љ

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"
_tf_keras_layer
І
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"
_tf_keras_layer
І
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
+О&call_and_return_all_conditional_losses
П__call__"
_tf_keras_layer
І
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"
_tf_keras_layer
І
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"
_tf_keras_layer
І
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"
_tf_keras_layer
І
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"
_tf_keras_layer
І
`	variables
atrainable_variables
bregularization_losses
c	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"
_tf_keras_layer
І
d	variables
etrainable_variables
fregularization_losses
g	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"
_tf_keras_layer
љ

hkernel
ibias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"
_tf_keras_layer
љ

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"
_tf_keras_layer
І
t	variables
utrainable_variables
vregularization_losses
w	keras_api
+†&call_and_return_all_conditional_losses
°__call__"
_tf_keras_layer
љ

xkernel
ybias
z	variables
{trainable_variables
|regularization_losses
}	keras_api
+Ґ&call_and_return_all_conditional_losses
£__call__"
_tf_keras_layer
©
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
+§&call_and_return_all_conditional_losses
•__call__"
_tf_keras_layer
√
Вkernel
	Гbias
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
+¶&call_and_return_all_conditional_losses
І__call__"
_tf_keras_layer
"
	optimizer
 "
trackable_list_wrapper
»
0
1
$2
%3
*4
+5
06
17
68
79
<10
=11
B12
C13
h14
i15
n16
o17
x18
y19
В20
Г21"
trackable_list_wrapper
»
0
1
$2
%3
*4
+5
06
17
68
79
<10
=11
B12
C13
h14
i15
n16
o17
x18
y19
В20
Г21"
trackable_list_wrapper
 "
trackable_list_wrapper
”
	variables
Иmetrics
Йlayer_metrics
trainable_variables
Кnon_trainable_variables
regularization_losses
Лlayers
 Мlayer_regularization_losses
э__call__
ь_default_save_signature
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
-
®serving_default"
signature_map
':%2board1/kernel
:2board1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 	variables
Нmetrics
Оlayer_metrics
!trainable_variables
Пnon_trainable_variables
"regularization_losses
Рlayers
 Сlayer_regularization_losses
€__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
':%2board2/kernel
:2board2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
&	variables
Тmetrics
Уlayer_metrics
'trainable_variables
Фnon_trainable_variables
(regularization_losses
Хlayers
 Цlayer_regularization_losses
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
):'2fileconv/kernel
:2fileconv/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
,	variables
Чmetrics
Шlayer_metrics
-trainable_variables
Щnon_trainable_variables
.regularization_losses
Ъlayers
 Ыlayer_regularization_losses
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
):'2rankconv/kernel
:2rankconv/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
2	variables
Ьmetrics
Эlayer_metrics
3trainable_variables
Юnon_trainable_variables
4regularization_losses
Яlayers
 †layer_regularization_losses
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
,:*2quarterconv/kernel
:2quarterconv/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
8	variables
°metrics
Ґlayer_metrics
9trainable_variables
£non_trainable_variables
:regularization_losses
§layers
 •layer_regularization_losses
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
*:(2largeconv/kernel
:2largeconv/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
>	variables
¶metrics
Іlayer_metrics
?trainable_variables
®non_trainable_variables
@regularization_losses
©layers
 ™layer_regularization_losses
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
':%2board3/kernel
:2board3/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
D	variables
Ђmetrics
ђlayer_metrics
Etrainable_variables
≠non_trainable_variables
Fregularization_losses
Ѓlayers
 ѓlayer_regularization_losses
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
H	variables
∞metrics
±layer_metrics
Itrainable_variables
≤non_trainable_variables
Jregularization_losses
≥layers
 іlayer_regularization_losses
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
L	variables
µmetrics
ґlayer_metrics
Mtrainable_variables
Јnon_trainable_variables
Nregularization_losses
Єlayers
 єlayer_regularization_losses
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
P	variables
Їmetrics
їlayer_metrics
Qtrainable_variables
Љnon_trainable_variables
Rregularization_losses
љlayers
 Њlayer_regularization_losses
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
T	variables
њmetrics
јlayer_metrics
Utrainable_variables
Ѕnon_trainable_variables
Vregularization_losses
¬layers
 √layer_regularization_losses
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
X	variables
ƒmetrics
≈layer_metrics
Ytrainable_variables
∆non_trainable_variables
Zregularization_losses
«layers
 »layer_regularization_losses
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
\	variables
…metrics
 layer_metrics
]trainable_variables
Ћnon_trainable_variables
^regularization_losses
ћlayers
 Ќlayer_regularization_losses
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`	variables
ќmetrics
ѕlayer_metrics
atrainable_variables
–non_trainable_variables
bregularization_losses
—layers
 “layer_regularization_losses
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
d	variables
”metrics
‘layer_metrics
etrainable_variables
’non_trainable_variables
fregularization_losses
÷layers
 „layer_regularization_losses
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 :
§А2dense/kernel
:А2
dense/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
j	variables
Ўmetrics
ўlayer_metrics
ktrainable_variables
Џnon_trainable_variables
lregularization_losses
џlayers
 №layer_regularization_losses
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_1/kernel
:@2dense_1/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
p	variables
Ёmetrics
ёlayer_metrics
qtrainable_variables
яnon_trainable_variables
rregularization_losses
аlayers
 бlayer_regularization_losses
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
t	variables
вmetrics
гlayer_metrics
utrainable_variables
дnon_trainable_variables
vregularization_losses
еlayers
 жlayer_regularization_losses
°__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
 :@ 2dense_2/kernel
: 2dense_2/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
z	variables
зmetrics
иlayer_metrics
{trainable_variables
йnon_trainable_variables
|regularization_losses
кlayers
 лlayer_regularization_losses
£__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
~	variables
мmetrics
нlayer_metrics
trainable_variables
оnon_trainable_variables
Аregularization_losses
пlayers
 рlayer_regularization_losses
•__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_3/kernel
:2dense_3/bias
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Д	variables
сmetrics
тlayer_metrics
Еtrainable_variables
уnon_trainable_variables
Жregularization_losses
фlayers
 хlayer_regularization_losses
І__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
(
ц0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
∆
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

чtotal

шcount
щ	variables
ъ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
ч0
ш1"
trackable_list_wrapper
.
щ	variables"
_generic_user_object
ё2џ
D__inference_model_1_layer_call_and_return_conditional_losses_6037425
D__inference_model_1_layer_call_and_return_conditional_losses_6037543
D__inference_model_1_layer_call_and_return_conditional_losses_6037194
D__inference_model_1_layer_call_and_return_conditional_losses_6037263ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ЋB»
"__inference__wrapped_model_6036360state"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
)__inference_model_1_layer_call_fn_6036697
)__inference_model_1_layer_call_fn_6037592
)__inference_model_1_layer_call_fn_6037641
)__inference_model_1_layer_call_fn_6037125ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
C__inference_board1_layer_call_and_return_conditional_losses_6037652Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_board1_layer_call_fn_6037661Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_board2_layer_call_and_return_conditional_losses_6037672Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_board2_layer_call_fn_6037681Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_fileconv_layer_call_and_return_conditional_losses_6037692Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_fileconv_layer_call_fn_6037701Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_rankconv_layer_call_and_return_conditional_losses_6037712Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_rankconv_layer_call_fn_6037721Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_quarterconv_layer_call_and_return_conditional_losses_6037732Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_quarterconv_layer_call_fn_6037741Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_largeconv_layer_call_and_return_conditional_losses_6037752Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_largeconv_layer_call_fn_6037761Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_board3_layer_call_and_return_conditional_losses_6037772Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_board3_layer_call_fn_6037781Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_layer_call_and_return_conditional_losses_6037787Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_flatten_layer_call_fn_6037792Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_flatten_1_layer_call_and_return_conditional_losses_6037798Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_1_layer_call_fn_6037803Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_flatten_2_layer_call_and_return_conditional_losses_6037809Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_2_layer_call_fn_6037814Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_flatten_3_layer_call_and_return_conditional_losses_6037820Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_3_layer_call_fn_6037825Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_flatten_4_layer_call_and_return_conditional_losses_6037831Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_4_layer_call_fn_6037836Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
р2н
F__inference_flatten_5_layer_call_and_return_conditional_losses_6037842Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_flatten_5_layer_call_fn_6037847Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
G__inference_dense_bass_layer_call_and_return_conditional_losses_6037858Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_dense_bass_layer_call_fn_6037868Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
∆2√
D__inference_dropout_layer_call_and_return_conditional_losses_6037873
D__inference_dropout_layer_call_and_return_conditional_losses_6037885і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Р2Н
)__inference_dropout_layer_call_fn_6037890
)__inference_dropout_layer_call_fn_6037895і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
м2й
B__inference_dense_layer_call_and_return_conditional_losses_6037906Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_layer_call_fn_6037915Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_1_layer_call_and_return_conditional_losses_6037926Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_1_layer_call_fn_6037935Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 2«
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037947
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037952і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ф2С
+__inference_dropout_1_layer_call_fn_6037957
+__inference_dropout_1_layer_call_fn_6037962і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_dense_2_layer_call_and_return_conditional_losses_6037973Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_2_layer_call_fn_6037982Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 2«
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037994
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037999і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ф2С
+__inference_dropout_2_layer_call_fn_6038004
+__inference_dropout_2_layer_call_fn_6038009і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
D__inference_dense_3_layer_call_and_return_conditional_losses_6038019Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_3_layer_call_fn_6038028Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 B«
%__inference_signature_wrapper_6037314state"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ђ
"__inference__wrapped_model_6036360Е$%BC<=6701*+hinoxyВГ6Ґ3
,Ґ)
'К$
state€€€€€€€€€
™ "1™.
,
dense_3!К
dense_3€€€€€€€€€≥
C__inference_board1_layer_call_and_return_conditional_losses_6037652l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Л
(__inference_board1_layer_call_fn_6037661_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€≥
C__inference_board2_layer_call_and_return_conditional_losses_6037672l$%7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Л
(__inference_board2_layer_call_fn_6037681_$%7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€≥
C__inference_board3_layer_call_and_return_conditional_losses_6037772lBC7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Л
(__inference_board3_layer_call_fn_6037781_BC7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€•
D__inference_dense_1_layer_call_and_return_conditional_losses_6037926]no0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
)__inference_dense_1_layer_call_fn_6037935Pno0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@§
D__inference_dense_2_layer_call_and_return_conditional_losses_6037973\xy/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ |
)__inference_dense_2_layer_call_fn_6037982Oxy/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€ ¶
D__inference_dense_3_layer_call_and_return_conditional_losses_6038019^ВГ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ~
)__inference_dense_3_layer_call_fn_6038028QВГ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "К€€€€€€€€€з
G__inference_dense_bass_layer_call_and_return_conditional_losses_6037858ЫрҐм
дҐа
ЁЪў
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€H
#К 
inputs/4€€€€€€€€€ј
"К
inputs/5€€€€€€€€€`
™ "&Ґ#
К
0€€€€€€€€€§
Ъ њ
,__inference_dense_bass_layer_call_fn_6037868ОрҐм
дҐа
ЁЪў
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
"К
inputs/3€€€€€€€€€H
#К 
inputs/4€€€€€€€€€ј
"К
inputs/5€€€€€€€€€`
™ "К€€€€€€€€€§§
B__inference_dense_layer_call_and_return_conditional_losses_6037906^hi0Ґ-
&Ґ#
!К
inputs€€€€€€€€€§
™ "&Ґ#
К
0€€€€€€€€€А
Ъ |
'__inference_dense_layer_call_fn_6037915Qhi0Ґ-
&Ґ#
!К
inputs€€€€€€€€€§
™ "К€€€€€€€€€А¶
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037947\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¶
F__inference_dropout_1_layer_call_and_return_conditional_losses_6037952\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ~
+__inference_dropout_1_layer_call_fn_6037957O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@~
+__inference_dropout_1_layer_call_fn_6037962O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@¶
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037994\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ¶
F__inference_dropout_2_layer_call_and_return_conditional_losses_6037999\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "%Ґ"
К
0€€€€€€€€€ 
Ъ ~
+__inference_dropout_2_layer_call_fn_6038004O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p 
™ "К€€€€€€€€€ ~
+__inference_dropout_2_layer_call_fn_6038009O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€ 
p
™ "К€€€€€€€€€ ¶
D__inference_dropout_layer_call_and_return_conditional_losses_6037873^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€§
p 
™ "&Ґ#
К
0€€€€€€€€€§
Ъ ¶
D__inference_dropout_layer_call_and_return_conditional_losses_6037885^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€§
p
™ "&Ґ#
К
0€€€€€€€€€§
Ъ ~
)__inference_dropout_layer_call_fn_6037890Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€§
p 
™ "К€€€€€€€€€§~
)__inference_dropout_layer_call_fn_6037895Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€§
p
™ "К€€€€€€€€€§µ
E__inference_fileconv_layer_call_and_return_conditional_losses_6037692l*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Н
*__inference_fileconv_layer_call_fn_6037701_*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€™
F__inference_flatten_1_layer_call_and_return_conditional_losses_6037798`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ В
+__inference_flatten_1_layer_call_fn_6037803S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€™
F__inference_flatten_2_layer_call_and_return_conditional_losses_6037809`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ В
+__inference_flatten_2_layer_call_fn_6037814S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€™
F__inference_flatten_3_layer_call_and_return_conditional_losses_6037820`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€H
Ъ В
+__inference_flatten_3_layer_call_fn_6037825S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€HЂ
F__inference_flatten_4_layer_call_and_return_conditional_losses_6037831a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ Г
+__inference_flatten_4_layer_call_fn_6037836T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€ј™
F__inference_flatten_5_layer_call_and_return_conditional_losses_6037842`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€`
Ъ В
+__inference_flatten_5_layer_call_fn_6037847S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€`®
D__inference_flatten_layer_call_and_return_conditional_losses_6037787`7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ А
)__inference_flatten_layer_call_fn_6037792S7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€ґ
F__inference_largeconv_layer_call_and_return_conditional_losses_6037752l<=7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ О
+__inference_largeconv_layer_call_fn_6037761_<=7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€ 
D__inference_model_1_layer_call_and_return_conditional_losses_6037194Б$%BC<=6701*+hinoxyВГ>Ґ;
4Ґ1
'К$
state€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ  
D__inference_model_1_layer_call_and_return_conditional_losses_6037263Б$%BC<=6701*+hinoxyВГ>Ґ;
4Ґ1
'К$
state€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ћ
D__inference_model_1_layer_call_and_return_conditional_losses_6037425В$%BC<=6701*+hinoxyВГ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ћ
D__inference_model_1_layer_call_and_return_conditional_losses_6037543В$%BC<=6701*+hinoxyВГ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ °
)__inference_model_1_layer_call_fn_6036697t$%BC<=6701*+hinoxyВГ>Ґ;
4Ґ1
'К$
state€€€€€€€€€
p 

 
™ "К€€€€€€€€€°
)__inference_model_1_layer_call_fn_6037125t$%BC<=6701*+hinoxyВГ>Ґ;
4Ґ1
'К$
state€€€€€€€€€
p

 
™ "К€€€€€€€€€Ґ
)__inference_model_1_layer_call_fn_6037592u$%BC<=6701*+hinoxyВГ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ґ
)__inference_model_1_layer_call_fn_6037641u$%BC<=6701*+hinoxyВГ?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€Є
H__inference_quarterconv_layer_call_and_return_conditional_losses_6037732l677Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Р
-__inference_quarterconv_layer_call_fn_6037741_677Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€µ
E__inference_rankconv_layer_call_and_return_conditional_losses_6037712l017Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Н
*__inference_rankconv_layer_call_fn_6037721_017Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€Є
%__inference_signature_wrapper_6037314О$%BC<=6701*+hinoxyВГ?Ґ<
Ґ 
5™2
0
state'К$
state€€€€€€€€€"1™.
,
dense_3!К
dense_3€€€€€€€€€