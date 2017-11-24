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

# c = Dict()
# fibonacci(20, c)
# print(c)


function gcd(m, n)
    if n == 0
        return m
    end
    if m < n
        return gcd(n, m)
    end
    return gcd(n, m % n)
end


print(gcd(13, 7))
