#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>


namespace rapidgzip
{
template<typename T>
struct Statistics
{
    constexpr
    Statistics() = default;

    constexpr
    Statistics( const Statistics& other ) = default;

    template<typename Container,
             typename = std::enable_if_t<!std::is_same_v<std::decay_t<Container>, Statistics<T> >, void> >
    constexpr explicit
    Statistics( const Container& values ) noexcept
    {
        for ( const auto value : values ) {
            merge( static_cast<T>( value ) );
        }
    }

    template<typename Iterator>
    constexpr
    Statistics( const Iterator& begin,
                const Iterator& end ) noexcept
    {
        for ( auto it = begin; it != end; ++it ) {
            merge( static_cast<T>( *it ) );
        }
    }

    [[nodiscard]] constexpr double
    average() const noexcept
    {
        return sum / count;
    }

    [[nodiscard]] constexpr double
    variance() const noexcept
    {
        /* Calculation makes use of expanded expression for the variance to avoid requiring there
         * average beforehand: Var(x) = < (x - <x>)^2 > = <x^2> - <x>^2
         * Furthermore, it is the sample variance, therefore divide by one count less for the outer average
         * because that degree of freedom has been used up for the sample average calculation. */
        return ( sum2 / count - average() * average() ) * count / ( count - 1 );
    }

    [[nodiscard]] constexpr double
    standardDeviation() const noexcept
    {
        return std::sqrt( variance() );
    }

    constexpr void
    merge( const Statistics& other ) noexcept
    {
        if ( other.count == 0 ) {
            return;
        }

        min = std::min( min, other.min );
        max = std::max( max, other.max );
        sum += other.sum;
        sum2 += other.sum2;
        count += other.count;
    }

    constexpr void
    merge( T value ) noexcept
    {
        min = std::min( min, value );
        max = std::max( max, value );
        sum += value;
        sum2 += std::pow( static_cast<double>( value ), 2 );
        ++count;
    }

    [[nodiscard]] std::string
    formatAverageWithUncertainty( bool    includeBounds = false,
                                  uint8_t sigmaMultiple = 1 ) const
    {
        const auto uncertainty = standardDeviation() * sigmaMultiple;
        /* Round uncertainty and value according to DIN 1333
         * @see https://www.tu-chemnitz.de/physik/PGP/files/Allgemeines/Rundungsregeln.pdf */

        /* Log10: 0.1 -> -1, 1 -> 0, 2 -> 0.301, 10 -> 1.
         * In order to scale to a range [0,100), we have to divide by 10^magnitude. */
        auto magnitude = std::floor( std::log10( uncertainty ) ) - 1;
        auto scaled = uncertainty / std::pow( 10, magnitude );

        /* Round uncertainties beginning with 1 and 2 to two significant digits and all others to only one.
         * This could probably be integrated into the magnitude calculation but it would be less readable. */
        if ( scaled >= 30 ) {
            magnitude += 1;
        }

        const auto roundToUncertainty =
            [magnitude] ( auto value )
            {
                return std::round( static_cast<double>( value ) / std::pow( 10, magnitude ) )
                       * std::pow( 10, magnitude );
            };

        /**
         * @note To be exact, we would also have to avoid trailing zeros beyond the certainty but that would require
         *       integrating unit formatting into this routine. E.g., do not write (13000 +- 1000) MB but instead
         *       (13 +- 1) GB.
         */
        std::stringstream result;
        result << std::fixed << std::setprecision( static_cast<int>( std::max( -magnitude, 0. ) ) );

        if ( includeBounds ) {
            result << roundToUncertainty( min ) << " <= ";
        }
        result << roundToUncertainty( average() ) << " +- " << roundToUncertainty( uncertainty );
        if ( includeBounds ) {
            result << " <= " << roundToUncertainty( max );
        }

        return result.str();
    }

public:
    /* Note that std::numeric_limits::infinity == 0 for integers. */
    T min{ std::numeric_limits<T>::max() };
    T max{ std::numeric_limits<T>::lowest() };

    double sum{ 0 };
    double sum2{ 0 };
    uint64_t count{ 0 };
};


template<typename T>
class Histogram
{
public:
    template<typename Container>
    explicit
    Histogram( const Container& container,
               uint16_t         binCount,
               std::string      unit = {} ) :
        m_statistics( container ),
        m_bins( binCount, 0 ),
        m_unit( std::move( unit ) )
    {
        if constexpr ( std::is_floating_point_v<T> ) {
            if ( !std::isfinite( m_statistics.min ) || !std::isfinite( m_statistics.max ) ) {
                return;
            }
        }

        if ( container.empty() ) {
            m_bins.clear();
            return;
        }

        if constexpr ( std::is_integral_v<T> ) {
            /* It seems almost impossible to me to fully avoid overflows here without casting to floating point. */
            const auto range = static_cast<double>( m_statistics.max ) - static_cast<double>( m_statistics.min );
            const auto usefulBinCount = static_cast<size_t>( range + 1.0 );
            if ( usefulBinCount < binCount ) {
                m_bins.resize( usefulBinCount, 0 );
            }
        }

        for ( const auto value : container ) {
            merge( static_cast<T>( value ) );
        }
    }

    Histogram( T           min,
               T           max,
               uint16_t    binCount,
               std::string unit = {} ) :
        m_statistics( std::array<T, 2>{ min, max } ),
        m_bins( binCount, 0 ),
        m_unit( std::move( unit ) )
    {
        if constexpr ( std::is_floating_point_v<T> ) {
            if ( !std::isfinite( m_statistics.min ) || !std::isfinite( m_statistics.max ) ) {
                return;
            }
        }

        if constexpr ( std::is_integral_v<T> ) {
            /* It seems almost impossible to me to fully avoid overflows here without casting to floating point. */
            const auto range = static_cast<double>( m_statistics.max ) - static_cast<double>( m_statistics.min );
            const auto usefulBinCount = static_cast<size_t>( range + 1.0 );
            if ( usefulBinCount < binCount ) {
                m_bins.resize( usefulBinCount, 0 );
            }
        }
    }

    constexpr bool
    merge( T value )
    {
        if constexpr ( std::is_floating_point_v<T> ) {
            if ( !std::isfinite( value ) ) {
                return false;
            }
        }

        if ( ( value < m_statistics.min ) || ( value > m_statistics.max ) || m_bins.empty() ) {
            return false;
        }

        const auto unitValue = static_cast<double>( value - m_statistics.min )
                               / static_cast<double>( m_statistics.max - m_statistics.min );
        const auto index =
            value == m_statistics.max
            ? m_bins.size() - 1U
            : static_cast<size_t>( std::floor( unitValue * m_bins.size() ) );

        m_bins.at( index ) += 1;
        return true;
    }

    [[nodiscard]] constexpr const auto&
    statistics() const noexcept
    {
        return m_statistics;
    }

    [[nodiscard]] constexpr double
    binStart( size_t binNumber ) const noexcept
    {
        return m_statistics.min + static_cast<double>( m_statistics.max - m_statistics.min )
               / static_cast<double>( m_bins.size() ) * static_cast<double>( binNumber);
    }

    [[nodiscard]] constexpr double
    binCenter( size_t binNumber ) const noexcept
    {
        return m_statistics.min + static_cast<double>( m_statistics.max - m_statistics.min )
               / static_cast<double>( m_bins.size() ) * ( static_cast<double>( binNumber ) + 0.5 );
    }

    [[nodiscard]] constexpr double
    binEnd( size_t binNumber ) const noexcept
    {
        return m_statistics.min + static_cast<double>( m_statistics.max - m_statistics.min )
               / static_cast<double>( m_bins.size() ) * ( static_cast<double>( binNumber ) + 1 );
    }

    [[nodiscard]] constexpr const auto&
    bins() const noexcept
    {
        return m_bins;
    }

    [[nodiscard]] std::string
    plot() const
    {
        if ( m_bins.empty() ) {
            return {};
        }

        std::stringstream result;

        std::vector<std::string> binLabels{ m_bins.size() };
        binLabels.front() = formatLabel( m_statistics.min );
        binLabels.back() = formatLabel( m_statistics.max );
        const auto maxBin = std::max_element( m_bins.begin(), m_bins.end() );
        for ( size_t i = 1; i < m_bins.size() - 1; ++i ) {
            if ( i == static_cast<size_t>( std::distance( m_bins.begin(), maxBin ) ) ) {
                binLabels[i] = formatLabel( binCenter( i ) );
            }
        }

        const auto maxLabelLength = std::max_element(
            binLabels.begin(), binLabels.end(),
            [] ( const auto& a, const auto& b ) { return a.size() < b.size(); } )->size();

        for ( size_t i = 0; i < m_bins.size(); ++i ) {
            const auto bin = m_bins[i];

            std::stringstream binLabel;
            binLabel << std::setw( maxLabelLength ) << std::right << binLabels[i];

            const auto binVisualSize = *maxBin == 0
                                       ? size_t( 0 )
                                       : static_cast<size_t>( static_cast<double>( bin )
                                         / static_cast<double>( *maxBin )
                                         * static_cast<double>( m_barWidth ) );
            std::stringstream histogramBar;
            histogramBar << std::setw( m_barWidth ) << std::left << std::string( binVisualSize, '=' );

            const auto countLabel = bin > 0 ? "(" + std::to_string( bin ) + ")" : std::string();

            result << binLabel.str() << " |" << histogramBar.str() << " " << countLabel << "\n";
        }

        return result.str();
    }

private:
    [[nodiscard]] std::string
    formatLabel( double value ) const
    {
        std::stringstream result;
        if ( std::round( value ) != value ) {
            result << std::scientific;
        }
        result << value;

        if ( !m_unit.empty() ) {
            result << " " << m_unit;
        }

        return result.str();
    }

private:
    const Statistics<T> m_statistics;
    std::vector<uint64_t> m_bins;

    const std::string m_unit;
    const uint16_t m_barWidth{ 20 };
};
}  // namespace rapidgzip
