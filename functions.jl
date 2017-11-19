using Formatting: printfmt


function fibonacci(n, c)
    if n in keys(c)
        return c[n]
    end
    if n < 2
        return n
    else
        printfmt("F($n)", n)
        r = fibonacci(n-1, c) + fibonacci(n-2, c)
        c[n] = r
        return r
    end
end

c = Dict()
fibonacci(20, c)
print(c)
