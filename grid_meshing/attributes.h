#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
#include <typeindex>
#include <iostream>
#include <vector>
#include <memory>
#include <map>
////////////////////////////////////////////////////////////////////////////////
namespace WA {//widow_Attribute

struct AttributeBase {
private:
	std::type_index m_DerivedType;
public:
	AttributeBase(std::type_index t) : m_DerivedType(t) { }
	virtual ~AttributeBase() = default;
	virtual void resize(size_t n) = 0;
	std::type_index type() const { return m_DerivedType; }
};

// -----------------------------------------------------------------------------

template<typename T>
struct Attribute : public AttributeBase {
	// Constructor
	Attribute(int rows) : AttributeBase(typeid(T)), content_(rows) { content_.setZero(); }

	// Resize data
	void resize(size_t r) { content_.resize(r); }

	// Data
	Eigen::Matrix<T, Eigen::Dynamic, 1> content_;
};

////////////////////////////////////////////////////////////////////////////////

// A generic class to manage attributes
class AttributeManager {

public:
	template<typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

private:
	size_t m_Size;
	std::map<std::string, std::shared_ptr<AttributeBase>> m_Attrs;

public:
	AttributeManager(size_t n = 0) : m_Size(n) { }

	// Number of attributes
	size_t size() const { return m_Size; }

	// Resize attributes
	void resize(size_t n) { for (auto ptr : m_Attrs) { ptr.second->resize(n); }; m_Size= n; }

	// Create a new attribute of type T
	template<typename T>
	Vector<T> & create(const std::string &name);

	// Retrieve attribute by name
	template<typename T>
	Vector<T> & get(const std::string &name);

	// Retrieve attribute by name
	template<typename T>
	const Vector<T> & get(const std::string &name) const;

	// Retrieve attribute type
	std::type_index type(const std::string &name) const { return m_Attrs.at(name)->type(); }

	// Retrieve attribute keys
	std::vector<std::string> keys() const;

	// Test whether a given attribute is present
	bool exists(const std::string &name) const { return m_Attrs.count(name) > 0; }
};

// -----------------------------------------------------------------------------

// Create a new attribute of type T
template<typename T>
AttributeManager::Vector<T> &
AttributeManager::create(const std::string &name) {
	if (m_Attrs.count(name)) {
		std::cerr << "[Attributes] Attribute [" << name << "] has already been created." << std::endl;
		return get<T>(name);
	} else {
		auto ptr = std::make_shared<Attribute<T>>(size());
		m_Attrs.emplace(name, ptr);
		return ptr->content_;
	}
}

// Retrieve attribute by name
template<typename T>
AttributeManager::Vector<T> &
AttributeManager::get(const std::string &name) {
	assert(m_Attrs.count(name) == 1);
	std::shared_ptr<AttributeBase> ptr = m_Attrs.at(name);
	auto * derived = dynamic_cast<Attribute<T> *>(ptr.get());
	assert(derived); //, "Incompatible type requested for attribute: " + name);
	return derived->content_;
}

// Retrieve attribute by name
template<typename T>
const AttributeManager::Vector<T> &
AttributeManager::get(const std::string &name) const {
	assert(m_Attrs.count(name) == 1);
	std::shared_ptr<const AttributeBase> ptr = m_Attrs.at(name);
	const auto * derived = dynamic_cast<const Attribute<T> *>(ptr.get());
	assert(derived); //, "Incompatible type requested for attribute: " + name);
	return derived->content_;
}

// -----------------------------------------------------------------------------

// Retrieve attribute keys
inline std::vector<std::string> AttributeManager::keys() const {
	std::vector<std::string> res;
	for (const auto & kv : m_Attrs) {
		res.emplace_back(kv.first);
	}
	return res;
}

}