#pragma once
#include <tuple>
#include <type_traits>

namespace type_checker {
namespace lambda_trait_detail {
template<class Ret, class Cls, class IsMutable, class... Args> struct types {
    using is_mutable = IsMutable;

    enum { arity = sizeof...(Args) };

    using return_type = Ret;

    template<size_t i> struct arg { typedef typename std::tuple_element<i, std::tuple<Args...>>::type type; };
};

template<typename T> struct lambda_type_trait_dispatch {};

template<class Ret, class Cls, class... Args> struct lambda_type_trait_dispatch<Ret (Cls::*)(Args...)> : types<Ret, Cls, std::true_type, Args...> {};

template<class Ret, class Cls, class... Args> struct lambda_type_trait_dispatch<Ret (Cls::*)(Args...) const> : types<Ret, Cls, std::false_type, Args...> {};

template<typename Lambda> struct lambda_trait : lambda_trait_detail::lambda_type_trait_dispatch<decltype(&Lambda::operator())> {};

}   // namespace lambda_trait_detail

/**
 * Some assertions on the filter lambda kernels types.
 */
template<typename func, int in_args> static constexpr void __host__ __device__ enforce_reqd_lambda_traits() {
    static_assert(in_args == 1 || in_args == 2);
    static_assert(std::is_same_v<typename lambda_trait_detail::lambda_trait<func>::is_mutable, std::false_type>, "I don't want mutable lambdas");
    static_assert(std::is_void_v<typename lambda_trait_detail::lambda_trait<func>::return_type>, "Lambda must have a void return type");
    static_assert(lambda_trait_detail::lambda_trait<func>::arity == in_args, "Wrong number of arguments");
    static_assert(std::is_same_v<typename lambda_trait_detail::lambda_trait<func>::template arg<0>::type, int>, "First arg must be int");
    if constexpr (in_args == 2) { static_assert(std::is_same_v<typename lambda_trait_detail::lambda_trait<func>::template arg<1>::type, int>, "Second arg must be int"); }
}
}   // namespace type_checker