#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, List, Any


def scan_directory(directory_path: str) -> Dict[str, Any]:
    """
    지정된 디렉토리를 스캔하여 연도별로 파일 목록을 정리합니다.
    saveData 폴더는 year/month 구조로 되어있습니다.

    Args:
        directory_path (str): 스캔할 디렉토리 경로

    Returns:
        Dict[str, Any]: 연도별로 정리된 파일 목록과 총 파일 수를 포함하는 딕셔너리
    """
    files_by_year: Dict[str, Dict[str, Any]] = {}
    total_files = 0

    # 디렉토리가 존재하는지 확인
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory_path}")

    # 연도 폴더 스캔
    for year_dir in os.listdir(directory_path):
        year_path = os.path.join(directory_path, year_dir)
        
        # 연도 폴더인지 확인
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            continue
            
        year = year_dir
        files_by_year[year] = {
            'count': 0,
            'files': []
        }
        
        # 월 폴더 스캔
        for month_dir in os.listdir(year_path):
            month_path = os.path.join(year_path, month_dir)
            
            # 월 폴더인지 확인
            if not os.path.isdir(month_path) or not month_dir.isdigit():
                continue
                
            # 월 폴더 내의 JSON 파일 스캔
            for file in os.listdir(month_path):
                if file.endswith('.json'):
                    file_path = os.path.join(month_path, file)
                    files_by_year[year]['files'].append(file_path)
                    files_by_year[year]['count'] += 1
                    total_files += 1

    return {
        'files_by_year': files_by_year,
        'total_files': total_files
    }


def save_to_json(data: Dict[str, Any], output_file: str) -> None:
    """
    데이터를 JSON 파일로 저장합니다.

    Args:
        data (Dict[str, Any]): 저장할 데이터
        output_file (str): 출력 파일 경로
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    """메인 함수"""
    try:
        # 현재 스크립트의 디렉토리 경로
        current_dir = Path(__file__).parent
        save_data_dir = current_dir / 'saveData'
        output_file = current_dir / 'mail_json_list.json'

        # 디렉토리 스캔
        result = scan_directory(str(save_data_dir))
        
        # JSON 파일로 저장
        save_to_json(result, str(output_file))
        
        print(f"파일 목록이 성공적으로 생성되었습니다: {output_file}")
        print(f"총 파일 수: {result['total_files']}")
        
        # 연도별 파일 수 출력
        print("\n연도별 파일 수:")
        for year, data in sorted(result['files_by_year'].items()):
            print(f"{year}년: {data['count']}개")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == '__main__':
    main()
