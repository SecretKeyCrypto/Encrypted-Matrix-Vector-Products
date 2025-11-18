#ifndef _DOCONTEXTIMPL_H_
#define _DOCONTEXTIMPL_H_

#include <unordered_map>
#include <typeindex>
#include <any>

class DoContextImpl {
    std::unordered_map<std::type_index, std::any> values;

public:
    // Put by copy
    template <typename T>
    T* put(const T& value) {
        values[std::type_index(typeid(T))] = value;
        return get<T>();
    }

    // Put by move
    template <typename T>
    T* put(T&& value) {
        values[std::type_index(typeid(T))] = std::any(std::forward<T>(value));
        return get<T>();
    }

    // Get (non-const) - returns pointer or nullptr
    template <typename T>
    T* get() {
        auto it = values.find(std::type_index(typeid(T)));
        if (it == values.end()) return nullptr;
        return std::any_cast<T>(&it->second);
    }

    // Get (const) - returns pointer or nullptr
    template <typename T>
    const T* get() const {
        auto it = values.find(std::type_index(typeid(T)));
        if (it == values.end()) return nullptr;
        return std::any_cast<T>(&it->second);
    }

    // Remove a value by type
    template <typename T>
    void remove() {
        values.erase(std::type_index(typeid(T)));
    }
};

#endif /* _DOCONTEXTIMPL_H_ */