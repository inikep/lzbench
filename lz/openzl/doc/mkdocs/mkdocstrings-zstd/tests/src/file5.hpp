#pragma once

#include <string>
#include <vector>

namespace ns5 {

/// @brief A function that does stuff
/// @param v A vector of integers
/// @return A vector of strings
std::vector<std::string> func5(const std::vector<int>& v);

class Class5 {
   public:
    using Type5 = int;

    /// A nested class
    struct NestedStruct5 {
        int member;
    };

    /// A nested enum
    enum class NestedEnum5 {
        VALUE1,
        VALUE2,
    };

    enum NestedWeakEnum5 { VALUE };

    /// Default constructor
    Class5() = default;
    /// Copy constructor
    Class5(const Class5&) = default;
    /// Move constructor
    Class5(Class5&&) = default;

    /// Copy assignment operator
    Class5& operator=(const Class5&) = default;
    Class5& operator=(Class5&&)      = default;

    /// An overloaded function
    void overload();
    /// An overloaded function that takes an int
    void overload(int i);

    /// A virtual function
    virtual void virtualFunc() = 0;

    /// A virtual function with a default implementation
    virtual void virtualFuncWithDefault() {}

    int publicInt;

   protected:
    int protectedInt;

   private:
    int privateInt;
};

void overloaded_func5();
int overloaded_func5(int x);

template <typename T>
void template_func5(T t);

template <>
void template_func5<int>(int t);

template <typename T, typename U>
class TemplateClass5 {};

template <typename T>
class TemplateClass5<T, int> {};

/**
 * @param x an int
 * @throws std::runtime_error if x is negative
 */
void func_that_throws(int x);

} // namespace ns5
