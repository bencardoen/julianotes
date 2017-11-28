# Julia notes

## Requirements

* Julia
* Jupyter
* IJulia (Pkg.add("IJulia"))

## Running
### Console
```bash
$ julia```
### Notebook
```bash
$ jupyter notebook
```
Inside notebook, shift+enter executes, esc switches to command mode. Tab autocompletes.

In command mode, typing m switches code to markdown

Shell commands can be executed by
```Julia
julia>;netstat -tulpna
```


#### Basics
##### Package management
```Julia
julia> Pkg.add("pname")
julia> Pkg.clone("repo.jl")
julia> Pkg.update()
julia> using pname # eq to from x import *
julia> import <packagename>: <name>
julia> include("filename") # executes script in filename
```
##### Help
```Julia
julia> ?<name>
```

##### Introspection
```Julia
julia> typeof(obj)
```
##### Type system
Conversion isn't as loose as in C[++]. E.g. bool->int is legal, the reverse is not.

```Julia
julia>if (1) # Not Legal
julia>if (Bool(1)) # Legal
```

Conversion
```Julia
julia> oftype(x,y) : converts y to type of x
```
Alias
```Julia
julia> typealias newname oldname
```

Parametric typing : use {T}
e.g.
```Julia
m = Array{Int64,1} # 1 dimensional array of 64 bit signed int
typeof(Array(Int64, 1)) == Vector{Int64} # true
```
Note expression{T} is a Type, not a value.
```Julia
Array(Int64, 1) == Vector{Int64} # false, lhs is not a type but an object
```

Numeric types:
Bool, Int{x}, Float{x} where x is [8,16,32,64], Complex{basetype}

Numeric operations
```Julia
julia> x = 5
julia> y = 7
julia> (3x + 5y) / y^2
julia> +(x,y)
julia> (x+y)y # legal
julia> (y)(x+y) # Error (parsing precedence)
julia> x += 1 # NOT in place, so x may even change type
```
*type{min|max}* provides range of types.

*epsilon(...)* provides mach eps

*{next|prev}float(x)*

```Julia
julia> 1.1 + 0.1 # 1.200002
julia> with_rounding(Type, Mode) do <code> end # Default is Float64, RoundNearest
```
*Big{Int|Float}*
```Julia
julia> b = BigInt(value)
julia> b = parse(BigInt,"123")
```
*{zero|one}(x)* Returns 1,0 of any x type.

*===* Tests if @lhs == @rhs

Strings
```Julia
julia> s = "42, surely? Or what do you think?"
# split(s[, delimiter] )
julia> s.replace(s, <orig>, <new> )
julia> s*s # concat
julia> strip(s) # remove whitespace
julia> match(<pattern>, <string> )
# returns matchobject
# pattern is of the form r"<regex>"
```
Hierarchy
```Julia
julia> Int64 <= Real # true
```
Define a method for the abstract type as a fallback function.

Defining types

```Julia
julia> type <name> end #  <name> should follow CamelCase
julia> n = <name>() # Default constructor is defined by env
type MyType
    a[::T] # force type
    b
    c
end
# defines a MyType as new type with a 3 arg constructor.
julia> m = MyType(1,2,3)
julia> m.a = 7
```
Parametric types
```Julia
type B{T <: Number} # Accepts any subtype T of Number (<: S is optional)
  a::T
end
julia> q = B{Int64}(7) #
julia> r = B(7) #
```
Note : fieldnames(<obj>) results in field names of object.

** // indicates Rational Number**

Singleton:
```Julia
type Singleton
end
```
Only one such an object can exist, all are references to this one object.
```Julia
julia> s = Singleton()
julia> q = Singleton()
julia> q===s
```
Immutable
```Julia
immutable Im
  field
  list
end
julia> im = Im(2, [1 2 3])# im's fields cannot change
julia> im.list = [2,3] # illegal
julia> im.list[1] = 5 # legal
```
Note: immutable objects are copied (pass by value, mutables are pass by reference)

Abstract types
```Julia
abstract Â«nameÂ» <: Â«supertypeÂ»
```


Const
```Julia
julia> const x = 7
julia> x = 9 # triggers a warning "redefining constant"
```
##### Scoping

global
```Julia
julia> x = 9 # implied global
function f()
    x = 8 # shadowing x, x here is a new variable, and local
    # if global x is intended, use:
    global x = 9
    # NOTE : once 'global x' is used, there is no more shadowing, e.g. a line x = 42 later on will still reference global x
end
```
local
```Julia
function f()
    i = 0
    for j= 1:10
      local i
      i = 7
    end
end
```

```Julia
julia>x = 0
julia>[ x for x in 1:3] # newly allocated x
julia>x == 0 # True
```
```Julia
julia>i=0
julia>for i = 1:3 # Not new i, reused i
      end
julia>i==3 #True // Same leaking as with Python
```
**Closures inherit scope of parent (lexical).**

##### Tuples
Similar to Python
```Julia
julia> m = "a", "b" # equiv to m=("a", "b")
julia> f, s = m # unpacking is supported
```

##### Arrays
```Julia
julia> m = [10, "10", true]
julia> m[1]
```
This is an Any-typed array. It's not legal to modify an array with an object not matching the expected type.
```Julia
julia> m = [1,2,3] # flat array (contrast with [1 2 3] which is a 1x3)
julia> m[1] = false # converts to 0
julia> m[1] = "false" # Error
julia> m = Array(<type>, <count>) #deprecated
julia> m = Array{type}(dims)
julia> m = fill(value, dims,)
julia> m = [1 2; 3 4] # 2x2 array
julia> m = [1;2;3;4] # NOT an 4x1 vector, flat array (4 element array)
julia> m = [1 2 3 4]' # Transpose of 1x4 = 4x1 array
```
Indices start from __1__
```Julia
julia> m = [1 2; 3 4]
julia> m[1] == 1
julia> m[1, 1] == 1
julia> m[1, :] == [1; 2]
julia> b = [true false; false true]
julia> m[b] == [1;4]
julia> m[2:3] = 42 # [1 42; 42; 4]
```
Operations
```Julia
julia> pop!(m) # ! indicated mutating method
julia> length(m) # all elements
julia> push!(m, 1)
julia> sort(m, rev=true) # reverse sorted copy of m
# Others include {max|min}imum , mean, std, var
```
Multidimensional (Vector = 1 dim, Matrix is n)
```Julia
julia> m = ones(2,3,4)
# 3 dimensional array of ones
julia> s = size(m)
# returns a tuple (2,3,4)
julia> s = ndims(m) # dimensions
julia> i = indices(m) # or indices(m, dim)
# returns tuple of valid indices (Base.OneTo(last)) element
julia> m[i:j] # slicing (returns [i,j] inclusive)
julia> m = [1,2,3,4]
julia> b = reshape(m,2,2) # b is a 2x2 representation of a, if you modify either you modify both
julia> b = b[:] # slice copy, destroys view gives original array
julia> b = squeeze(b, dim)[:] #
```
Matrix Operations
```Julia
julia> c = [1 0 ; 0 1]
julia> e = c' # transpose
julia> i = inv(c)
julia> b = [1 2 ; 3 4]
julia> a = [1 1 ; 1 1]
julia> b * a
#a x = b, solve for x
julia> a \ b
julia> dot(ones(3), ones(3))
julia> ones(2) * a # dim error (2x1 x 2x2)
julia> a * ones(2) # ok
julia> a .* 5 # element wise mult (.x where x is operator is element wise as in Matlab)
julia> 5a == a.*5 # scalar mult is unambiguous
julia> a.>b # element wise comparison (boolean matrix as result)
julia> a[a.>0] # get all elements larger than zero
julia> log(a) # vectorized log of a, element wise, equal to [log(x) for x in a]
```
Linear algebra
```Julia
julia> {det|trace|eigvals|rank}(A)
```

_end_ keyword in a sequence reference last element

###### Dicts
```Julia
julia> a = Dict(key => value [, key => value])
julia> a[key] # gets value
for d in a
    println(d.second) # value (first is key)
end
```
Mutating is done by accessor
```Julia
for d in keys(a)
    a[d] = <newvalue>
end
```
Iterating over a and modifying pair in place will lead to an error.

##### Control flow
###### For
```Julia
for <vname> in begin:end
    <operate on vname>
end
# begin:end is an generator, not a sequence
for <vname> in collection
  <operate on vname>
end
# advanced ranges
for j in start:end, k in start2:end2
    <stmts>
end
# e.g.
for j in 1:2, k in 1:3
    println(j, k)
end
# results in 11, 12, 13, 21, 22, 23
```
###### While
```Julia
while <condition>
    < body >
    # break, continue allowed
end
```

###### If
Not a new scope !!
```Julia
if <condition>
    < body >
[elsif <condition>]
[else]
end
```

###### Exceptions
```Julia
julia> throw(<exception instance>) # raise Except
julia>
```
Info/Logging
```Julia
julia> {info|warning|error}("z") # error raises error and logs.
```
```Julia
try
<stmts>
[catch [excp] # first symbol after catch is name exception, if anonymous use ;
<stmts>
]
[finally
<stmts> # executed regardless of exception
]
end
```
###### Compound expressions
Idea : have a series of expressions executed, the result of the last is returned (as in C)
```Julia
julia> a =begin
          x=1+2
          y=3
          x-y
        end
```
Or single line variant
```Julia
julia> a = (x=1+2; y=3; x-y)
julia> a = begin x=1+2; y=3; x-y end
```
###### Coroutines
Producer : f - object that calls produce(value)
Consumer : f - object that calls consume(Task)
```Julia
function exprod()
    produce("first")
    for n=1:10
        produce(n)
    end
    produce("last")
end
```

```Julia
julia> p = Task(exprod)
julia> consume(p)
julia> for x in p
          println(p)
       end
```
Single call usage, but equally effective as an iterator. Note that the function object is resumed on call, returning several times with the _same_ state.
These not 12 invocations of the function with a clear function stack frame, but a resuming of the same function in the stack frame.
Without catch, the statement resolves to nothing.

Task expects a function without parameters, to give a producer parameters use :
```Julia
function paramtask(arg)
<stmts>
end
wrapper = Task(() -> paramtask(param))
||
wrapper = @task paramtask(param)
```

##### Parallel computing
Note : start julia with *julia -p <processcount>* or in a notebook *addprocs<processcount>*
Other means of controlling cluster : *rmprocs*, *workers* (list of pids), *nworkers()* returns process count, *procs(int)* list of pids registered for physical id=intm *procs(SharedArray)* pids sharing array, *procs* list of pids
Based on remote reference (Future, RemoteChannel) and remote call. Remote call returns a Future, extract value by (fetch() or wait() for immediate retrieval).

```Julia
julia> f = remotecall(<callable>, processindex (1...), arg1, arg2, ...)
# f.id, f.v (value), f.whence (called by), f.where (executed at)
julia> fetch(f) # get value
julia> wait(f) # wait for computation explicitly
julia> remotecall_fetch(<function>, processindex, arg1, arg2, ...) # run and get, faster than fetch(rcall(...))
julia> @spawnat <processindex> <expression>
# returns future object resulting from expression executed at processindex
julia> @spawn <expression>
# return future object resulting from expression, process indices resolved automatically
# Processes don't have automatic scope sharing
@everwhere function <name>(args)
    <stmts>
end
#
@everywhere x = existingfunction()
f = remotecall_fetch(()->x, 2)
```
# Take care not to move data when not needed (obvious, but still)
```Julia
a = [x for x in 1:100 ] # local variable a in process 1
f = @spawn mean(a) # moves a to process 2 (or other), then calculates result
f2 = @spawn mean([x for x in 1:100]) # array is created on process2 and processed there, eliding copy
```

Parallel computation
```Julia
# Parallel for
# [<name> =] @parallel <operator> for <range expression>
#     <expression> # Value of expression is argument to operator
# end
# e.g. ^^ is a parallel sum (ENSURE operator is associative)
# entire expression returns computation result
nheads = @parallel (+) for i = 1:200000000
    UInt(rand(Bool)) # This local variable is local per process
end
#
a = zeros(10)
@parallel for i = 1:10
    #println(a[i])
    a[i] = i
    #println(a[i])
end
# Fails to initialize a in process 1, each has a local copy of a, not broadcasted.
b = SharedArray(Float64, 10)
@parallel for i= 1:10
    b[i] = i
end
# without operator, the tasks run async (i.e. return future without waiting on each other)
# with operator, the results have to be combined
# Optionally, use @sync @parallel to explicitly wait for each task
# Concept here is reduction, have x = f(x, v[i]) : accumulate result of f(x,v) in x.
# last step (summing results of tasks is done on calling process)
```

Use @parallel for for large counts of small time consuming function calls. (e.g. summing array)
Use pmap for expensive (item wise) function calls)

```Julia
# Parallel Map
# pmap(<function>, object)
M = Matrix{Float64}[rand(1000,1000) for i=1:10] # A 10 element matrix where each element is a 1000x1000 random matrix
a=pmap(svd, M)
## optionally, use @sync, @async if needed for dynamic scheduling (todo)
```

A channel is more powerful, writeable exchange.

```Julia
@everywhere function removeprocs()
    for n in procs()
        if n>1
            rmprocs(n)
        end
    end
end

if nprocs() < 2
    addprocs(5)
end
# basic channel example
worker_pool = Base.default_worker_pool()
channels = [RemoteChannel() for i in 1:nworkers()]
for (i, c) in enumerate(channels)
    @show put!(c, i)
end

for c in channels
    @show take!(c)
end

removeprocs()
```

```Julia
# Example from the docs showing that colluding writes/reads will lead to performance regression
# This function returns the (irange,jrange) indexes assigned to this worker
@everywhere function myrange(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0, 1:0
    end
    nchunks = length(procs(q)) # all processes referencing q
    splits = [round(Int, s) for s in linspace(0,size(q,2),nchunks+1)]
    1:size(q,1), splits[idx]+1:splits[idx+1]
end


@everywhere function advection_chunk!(q, u, irange, jrange, trange)
    @show (irange, jrange, trange)  # display so we can see what's happening
    for t in trange, j in jrange, i in irange
        q[i,j,t+1] = q[i,j,t] +  u[i,j,t]
    end
    q
end

@everywhere advection_shared_chunk!(q, u) = advection_chunk!(q, u, myrange(q)..., 1:size(q,3)-1)

@everywhere advection_serial!(q, u) = advection_chunk!(q, u, 1:size(q,1), 1:size(q,2), 1:size(q,3)-1)

@everywhere function advection_parallel!(q, u)
    for t = 1:size(q,3)-1
        @sync @parallel for j = 1:size(q,2) # @sync : force tasks to wait on each other
            for i = 1:size(q,1)
                q[i,j,t+1]= q[i,j,t] + u[i,j,t]
            end
        end
    end
    q
end

@everywhere function advection_shared!(q, u)
    @sync begin
        for p in procs(q)
            @async remotecall_wait(advection_shared_chunk!, p, q, u) # async, since there is no longer an interdependency.
        end
    end
    q
end
```

Shared data (arrays)
```Julia
addprocs(4)
# Create a shared array, 3x4, initialize with process index, localindexes splits indices over range.
# init=<function>, where function = f(S::SharedArray)
s = SharedArray(UInt, (3,4), init= s -> s[Base.localindexes(s)] = myid())
# Get underlying data
q = sdata(s)
removeprocs()
```
Note : ! races if any item is shared

###### Threads (experimental)
```Julia
export JULIA_NUM_THREADS=4
```
```Julia
Threads.nthreads() # see ^
a = zeros(10)
Threads.:threads for i = 1:10
    a[i] = Threads.threadid()
end
```
Results in a initialized by 4 threads, without races.

##### Functions
```Julia
function[!] <name>(<args>)
  <body>
  [return [value]]
end
```
_nothing_ is eq to void (C), None(Python)

**!** Denotes that f is mutating on the arguments, without it arguments are read only. E.g.. for a list this does not mean that the elements can't be changed, only that the object can't be changed.

**Return** Is optional, the last expression of the function body is the return value by default, return can be used for more cases.

Shorthand Functions
```Julia
julia> f(x) = x^2
julia> x-> x^2
```
Use unnamed functions (l's)
```Julia
julia> map(x->x^2, collection) #use map if result is needed
julia> foreach(f, c) #call f on each in c, discard value
```
Arguments: same rules as in Python.
```Julia
function f(first, second; third=named, fourth[::Type]=named )
function f(x; keywords...) # slow
```
**Note** A default argument does not imply a keyword argument. Scoping rules for keyword arguments are local (in argument space) whereas for defaults they are left to right
```Julia
function f(first, second=3; third=named)
```

Multiple dispatch
```Julia
julia> @which <functioncall> # returns exact method being executed
```
Method is function with certain set of arguments.

E.g. function \*(T,T), method \*(Int64, Int64)

Multiple dispatch : with statement f(S, T), lookup methods of f matching S,T.
If none is found, match super(S), super(T)

Defining typed Functions
```Julia
function f(x::Int64)
    println("Integer 64 version called")
end
function f(x)
    println("Generic version called")
end

f("abc") # calls generic version
f(42) # calls Int64 version
```
**Vectorization**
```Julia
sin.(cos.([1,2,3]))
```
<f>. = vectorize f on argument, that is,

##### IO
```Julia
julia> f = open(<fname>, mode) # mode is w,r
julia> write(f, <content>)
julia> readall(f)
julia> for q in eachline(f) println(q) end
```

##### Iterators
```Julia
julia> a = Dict(1=>2, 2=>1)
julia> q = collect(keys(a))
```
Collect transforms iterator to collection.

```Julia
julia> a = [1,2,3]
julia> s=start(a) # starting iterator state
julia> next(a, s) # (current element, next iter)
julia> done(a, s) # check if end is reached
julia> rest(a, 4) # iterator starting at 4th (in this case, state index)
julia> a = [x for x=10:20]
julia> b = [(index, element) for (index, element) in enumerate(a)]
julia> a5 = take(a, 5)
julia> l5 = drop(a, 5) # generates all but first 5
julia> cycle(a) # infinite cycle
julia> repeated(x[, n]) # infinite loop of x, or n times x
julia> countfrom(start, step) # infinite series
```


```Julia
julia> a= [1,2,3]
julia> b = [3,4,5]
julia> c, d, e = zip(a,b)
julia> b1, b2, b3 = b... # Unzip
```
List comprehension (pythonic)
```Julia
julia> a = [1,2,3]
julia> b = [i for i in a if i % 2 == 0]
```

##### Collections
```Julia
isempty(c)
empty!(c) # clear c
length(c)
endof(c) -> last index
```
Membership test
```Julia
2 in a# compares by == for lists,isequal on key for dictitems
2 in keys(a) # dicts
haskey(a, 2) #
eltype(a)
findin(a, b) #indices of a in b
collect(a) # equivalent of a.items()
```
Operations
```Julia
unique(iter[, dim]) # returns unique set in iter, in order of passing
reduce(op, neutral, iter) #neutral is neutral element operator (if empty), op is binary associative operator
extrema(a) # min, max
ind{min|max}(a)
find{min|max}(iter) # value, index
foreach(f, c) # apply f for x in c, discarding result
map(f, c) # get result of f on each c
map!(f,result, iterable) # apply f on iterable storing in result
mapreduce(f, op, neutral, iter) # reduce( op(neutral, f(iter)) )
broadcast(f, l, r) # Used when you apply f(l, r) but l and r are not same dimensions (e.g. vector + matrix), without consuming more memory
```

###### Use case : list of tuples to tuple of lists
```Julia
function unzip(lst)
    Nt = length(lst[1])
    lsts = [[] for _ in 1:Nt]
    for el in lst
        for index in 1:Nt
            push!(lsts[index],el[index])
        end
    end
    return lsts
end

function unzipr(lst)
    return reduce( ([],[]), lst) do x,y; for i in 1:length(y) push!(x[i], y[i]) end;x;end;
end
```

##### Logic
Logical and/or/not  (bool only) : &&, ||, !
Bitwise : &, |, ~ (watch out for signedness UInt8 != Int8)


##### Performance
Fully specifying type enhances performance
```Julia
julia> x = [rand() for x in 1:1e5]
function f(x)
    s = 0
    for y in x
        s += y
    end
end
function f(x::Array{Float64,1}) # typed method for float arrays
    s = 0
    for y in x
        s += y
    end
end
julia> @time g(x)
julia> @time g(x) # Could be faster, but only if Julia isn't already able to infer type.
julia> x = Any[rand() for x in 1:1e5] #
julia> g(x) # Calls method with generic parameters. No type inference can be made, so no optimizations possible.
```
```Julia
type B
     a # Impossible to infer typeof(a)
end
```
Since the type of a can only be known at runtime, performance degrades. If a's Type is known not based on its runtime value, performance is high.

If B is genericly typed with T, and field a as well, all depends on T. If T is an Abstract type, explicit, then it can still change at runtime
(e.g Int64 v Int32 as subtypes of Numeric).
```Julia
type B{T <: Number}
    a::T
end
julia> b = B{Int64}(43) # typeof(b.a is Int64), type is set not by value at runtime, so opt is possible
```

Inspecting generated code
```Julia
function f(x)
    x+=1
code_llvm(f,(B{Int64},))
code_llvm(func,(B{Number},))
code_llvm(func,(B,))
```

For the same reason (type inference), avoid global variables.


Important : on first call, a function will be compiled. So benchmarking with @time
should either split that call, or use an average.

Promote inlining by splitting aggregate function into smaller functions.

Use zero(), one() to create type stable expressions

```Julia
a = x < 0 ? zero(x) : x # x is Any Typed, so without zero(x) this expression can return 2 types
```
Similar, avoid type changing of variables
```Julia
julia> x = 8 # Int64
julia> x = x/5 # Float64
```

Split function in functions where argument type is known:
If type variable x is known after y statements, split remaind of code into separate function to allow optimizations.

#### Python interface
using PyCall
@pyimport math
math.sin(math.pi / 4) - sin(pi / 4)  # returns 0.0


#### Final Notes
Memory access is column major !!
